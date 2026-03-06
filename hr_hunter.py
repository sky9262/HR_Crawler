#!/usr/bin/env python3
"""
Crawl4AI Direct HR Hunter
Uses Crawl4AI without proxy to crawl company websites directly
"""

import pandas as pd
import json
import re
import os
import sys
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
from dotenv import load_dotenv
from datetime import datetime

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

load_dotenv()

TEST_MODE = os.getenv('TEST_MODE', 'false').lower() == 'true'
TEST_JOB_COUNT = int(os.getenv('TEST_JOB_COUNT', '3'))

# Gemini API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')

# Ollama Local LLM Configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:3b')


@dataclass
class HRContact:
    company: str
    company_url: str
    hr_name: str
    title: str
    email: str
    email_type: str
    source: str
    confidence: float


class Crawl4AIDirectHunter:
    """HR Hunter using Crawl4AI without proxy"""
    
    def __init__(self):
        self.crawler = None
    
    async def init_crawler(self):
        """Initialize Crawl4AI without proxy"""
        browser_config = BrowserConfig(
            headless=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        self.crawler = AsyncWebCrawler(config=browser_config)
        await self.crawler.start()
    
    async def close(self):
        if self.crawler:
            await self.crawler.close()
    
    def is_email_related_to_company(self, email: str, company: str, company_url: str = "") -> bool:
        """Check if email domain is related to the target company dynamically"""
        if not email or '@' not in email:
            return False
        
        email_domain = email.split('@')[1].lower()
        email_user = email.split('@')[0].lower()
        
        # Extract company name variations
        company_lower = company.lower()
        
        # Remove common suffixes and extract core name
        for suffix in ['株式会社', 'inc.', 'corp.', 'corporation', 'limited', 'ltd.', 'co.,', 'co.', 'kk', '合同会社']:
            company_lower = company_lower.replace(suffix, '').strip()
        
        # Extract domain from company URL if available
        expected_domains = []
        if company_url:
            from urllib.parse import urlparse
            try:
                parsed = urlparse(company_url)
                if parsed.netloc:
                    domain = parsed.netloc.lower()
                    # Remove www. prefix
                    if domain.startswith('www.'):
                        domain = domain[4:]
                    expected_domains.append(domain)
                    # Also add without subdomain
                    if '.' in domain:
                        main_domain = '.'.join(domain.split('.')[-2:])
                        expected_domains.append(main_domain)
            except:
                pass
        
        # Build company name variations for domain matching
        company_variations = [
            company_lower,
            company_lower.replace(' ', ''),
            company_lower.replace(' ', '-'),
            company_lower.replace('.', '')
        ]
        
        # Add romaji variations for Japanese companies
        import re
        # Extract alphabetic parts that might be romaji
        romaji_parts = re.findall(r'[a-zA-Z]+', company)
        for part in romaji_parts:
            if len(part) > 2:
                company_variations.append(part.lower())
        
        # STRICT CHECK 1: If we have expected domain from company URL, prioritize it
        if expected_domains:
            for expected in expected_domains:
                if email_domain == expected or email_domain.endswith('.' + expected):
                    return True
        
        # STRICT CHECK 2: Email domain should contain company name
        for variation in company_variations:
            if len(variation) >= 3:  # Minimum length to avoid false matches
                # Check if variation is in domain
                if variation in email_domain:
                    return True
                # Check common TLD variations
                for tld in ['.jp', '.com', '.co.jp', '.net', '.org']:
                    if email_domain == variation + tld or email_domain.endswith(variation + tld):
                        return True
        
        # STRICT CHECK 3: For generic email providers, be more restrictive
        free_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com', 'me.com', 'yahoo.co.jp']
        if email_domain in free_providers:
            # Only accept if the username contains company name
            for variation in company_variations:
                if len(variation) >= 3 and variation in email_user:
                    return True
            # If no company name in username, reject free email providers
            return False
        
        # STRICT CHECK 4: Reject unrelated company domains dynamically
        # Build list of domains that contain company name but are NOT the target
        if len(company_lower) >= 3:
            # Common TLDs to check
            tlds = ['.com', '.co.jp', '.jp', '.net', '.org', '.co.uk', '.io', '.tech']
            # Common suffixes that might be added to company names
            suffixes = ['health', 'food', 'digital', 'tech', 'mail', 'soft', 'systems', 'solutions', 'group', 'global']
            
            for tld in tlds:
                for suffix in suffixes:
                    unrelated_domain = f"{company_lower}{suffix}{tld}"
                    if email_domain == unrelated_domain:
                        return False
                    # Also check with hyphen
                    unrelated_domain_hyphen = f"{company_lower}-{suffix}{tld}"
                    if email_domain == unrelated_domain_hyphen:
                        return False
        
        # DEFAULT: Reject if doesn't match any criteria
        return False
    
    async def validate_contact_with_llm(self, contact: HRContact, company: str) -> tuple[bool, str]:
        """
        Use Ollama LLM to validate if a contact is genuinely related to the target company.
        Returns (is_valid, reason)
        """
        import aiohttp
        
        prompt = f"""You are a data validation assistant. Analyze this contact information and determine if it is genuinely related to the target company.

Target Company: {company}
Contact Name: {contact.hr_name}
Contact Email: {contact.email}
Contact Title: {contact.title}
Source URL: {contact.company_url}

Task:
1. Determine if this contact appears to be a real HR person/recruiter at the target company
2. Check if the email domain is appropriate (company domain, job board domain, or recruiter domain)
3. Consider that Japanese companies may use various email formats and domains
4. Job board emails (like @jobs.mynavi.jp, @en-japan.com) are VALID for job postings
5. LinkedIn URLs are VALID sources

Respond in JSON format:
{{
    "is_valid": true/false,
    "reason": "brief explanation of your decision",
    "confidence": "high/medium/low"
}}

Important:
- Be lenient with job board emails (they are valid for job postings)
- Consider that some companies use parent company domains
- Free email providers (gmail, yahoo) are acceptable if the person is verified
- When in doubt, mark as valid with medium confidence"""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1}
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get('response', '')
                        
                        # Extract JSON from response
                        try:
                            # Look for JSON in the response
                            json_match = re.search(r'\{[^}]*"is_valid"[^}]*\}', response_text, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                                validation = json.loads(json_str)
                                is_valid = validation.get('is_valid', True)
                                reason = validation.get('reason', 'No reason provided')
                                return is_valid, reason
                        except:
                            pass
                        
                        # If JSON parsing fails, check for keywords
                        response_lower = response_text.lower()
                        if '"is_valid": false' in response_lower or 'not valid' in response_lower:
                            return False, "LLM determined contact is not valid"
                        
                        return True, "LLM validation passed"
                    
                    return True, "LLM unavailable, accepting by default"
                    
        except Exception as e:
            return True, f"LLM validation error: {str(e)}, accepting by default"
    
    async def filter_results_with_llm(self, contacts: List[HRContact], company: str) -> List[HRContact]:
        """
        Filter all contacts using LLM validation instead of pattern matching.
        """
        print(f"\n  🤖 LLM Filtering {len(contacts)} contacts for {company}...")
        
        validated_contacts = []
        
        for contact in contacts:
            # Quick pre-filter: remove obvious garbage
            if not contact.hr_name or len(contact.hr_name) < 2:
                continue
            if contact.email and ('example' in contact.email or 'yourmail' in contact.email):
                continue
            
            # LLM validation
            is_valid, reason = await self.validate_contact_with_llm(contact, company)
            
            if is_valid:
                validated_contacts.append(contact)
                print(f"    ✓ Validated: {contact.hr_name[:25]} - {contact.email[:30] if contact.email else 'no email'}")
            else:
                print(f"    ✗ Filtered out: {contact.hr_name[:25]} - Reason: {reason}")
        
        print(f"  ✅ LLM kept {len(validated_contacts)}/{len(contacts)} contacts")
        return validated_contacts
    
    def save_contact_immediately(self, contact: HRContact, company: str = ""):
        """Save a contact immediately to CSV (append mode) with relaxed validation"""
        
        # Relaxed validation: only skip obviously fake emails
        if contact.email:
            email_lower = contact.email.lower()
            # Skip obviously fake/example emails
            if 'example' in email_lower or 'yourmail' in email_lower or 'test@' in email_lower:
                print(f"      ⚠️ Skipping fake email: {contact.email}")
                return
            # Skip invalid email format
            if '@' not in contact.email or '.' not in contact.email.split('@')[1]:
                print(f"      ⚠️ Skipping invalid email: {contact.email}")
                return
        
        output_path = Path(".")
        csv_path = output_path / "crawl4ai_direct_summary.csv"
        
        # Create DataFrame from single contact
        df_new = pd.DataFrame([asdict(contact)])
        
        # If file exists, append; otherwise create new
        if csv_path.exists():
            df_new.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_new.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"      💾 Saved: {contact.hr_name[:20]} - {contact.email[:25] if contact.email else 'no email'}")
    
    async def filter_and_format_results(self, contacts: List[HRContact], company: str = "") -> List[HRContact]:
        """
        Use LLM-based filtering to clean and format final results.
        Replaces pattern-based filtering with intelligent LLM validation.
        """
        print("\n  🤖 Filtering and formatting results with LLM...")
        
        if not contacts:
            return []
        
        # Step 1: Quick pre-filter - remove obvious garbage only
        pre_filtered = []
        for c in contacts:
            # Skip if no name or name is too short
            if not c.hr_name or len(c.hr_name) < 2:
                continue
            # Skip obviously fake emails
            if c.email and ('example' in c.email.lower() or 'yourmail' in c.email.lower()):
                continue
            pre_filtered.append(c)
        
        print(f"     Pre-filtered: {len(contacts)} → {len(pre_filtered)} contacts")
        
        # Step 2: LLM-based validation for each contact
        if company and pre_filtered:
            validated = await self.filter_results_with_llm(pre_filtered, company)
        else:
            validated = pre_filtered
        
        # Step 3: Deduplicate by email (keep highest confidence)
        email_map = {}
        for c in validated:
            if c.email:
                key = c.email.lower()
                if key not in email_map or c.confidence > email_map[key].confidence:
                    email_map[key] = c
        
        # Step 4: Deduplicate by name (keep highest confidence)
        name_map = {}
        for c in validated:
            if not c.email:  # Only for contacts without email
                key = c.hr_name.lower()
                if key not in name_map or c.confidence > name_map[key].confidence:
                    name_map[key] = c
        
        # Combine results
        final_results = list(email_map.values()) + list(name_map.values())
        
        # Step 5: Sort by confidence (highest first)
        final_results.sort(key=lambda x: x.confidence, reverse=True)
        
        print(f"     Final results: {len(final_results)} contacts")
        return final_results
    
    def save_url_for_later(self, company: str, url: str, title: str):
        """Save URL to urls.json for later crawling"""
        output_path = Path(".")
        urls_file = output_path / "urls.json"
        
        # Load existing data
        if urls_file.exists():
            with open(urls_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
        
        # Add URL to company list
        if company not in data:
            data[company] = []
        
        url_entry = {
            'url': url,
            'title': title,
            'found_at': datetime.now().isoformat()
        }
        
        # Avoid duplicates
        if not any(u['url'] == url for u in data[company]):
            data[company].append(url_entry)
            with open(urls_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    def save_name_for_later(self, company: str, name: str, source: str):
        """Save name to names.json for later deep search"""
        # Strip whitespace from name
        name = name.strip()
        
        output_path = Path(".")
        names_file = output_path / "names.json"
        
        # Load existing data
        if names_file.exists():
            with open(names_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
        
        # Add name to company list
        if company not in data:
            data[company] = []
        
        name_entry = {
            'name': name,
            'source': source,
            'found_at': datetime.now().isoformat()
        }
        
        # Avoid duplicates (case insensitive comparison)
        if not any(n['name'].lower() == name.lower() for n in data[company]):
            data[company].append(name_entry)
            with open(names_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"    💾 Saved name for later: {name}")
    
    def save_email_for_later(self, company: str, email: str, source: str, company_url: str = ""):
        """Save email to emails.json with company validation"""
        # Validate email belongs to target company
        if not self.is_email_related_to_company(email, company, company_url):
            print(f"      ⚠️ Skipping email: {email} does not appear related to {company}")
            return
        
        output_path = Path(".")
        emails_file = output_path / "emails.json"
        
        # Load existing data
        if emails_file.exists():
            with open(emails_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
        
        # Add email to company list
        if company not in data:
            data[company] = []
        
        email_entry = {
            'email': email,
            'source': source,
            'found_at': datetime.now().isoformat()
        }
        
        # Avoid duplicates
        if not any(e['email'] == email for e in data[company]):
            data[company].append(email_entry)
            with open(emails_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    async def analyze_with_gemini(self, company: str, names: List[str], emails: List[str], urls: List[str]) -> Dict:
        """Use Gemini AI to analyze HR data and provide insights"""
        if not GEMINI_API_KEY:
            print("    ⚠️ Gemini API key not found, skipping AI analysis")
            return {}
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            
            # Prepare the prompt
            prompt = f"""Analyze the following HR contact data for {company}:

NAMES FOUND:
{chr(10).join(f"- {name}" for name in names) if names else "No names found"}

EMAILS FOUND:
{chr(10).join(f"- {email}" for email in emails) if emails else "No emails found"}

URLS FOUND:
{chr(10).join(f"- {url}" for url in urls[:20]) if urls else "No URLs found"}

Please provide:
1. Which names are most likely to be HR personnel (rank by likelihood)
2. Which emails are most likely to be HR emails (rank by likelihood)
3. Which URLs are most likely to contain HR contact information
4. Suggestions for additional searches to find more HR contacts
5. Any patterns or insights you notice

Respond in JSON format with these keys: likely_hr_names, likely_hr_emails, likely_hr_urls, search_suggestions, insights"""

            print(f"    🤖 Analyzing data with Gemini AI...")
            response = model.generate_content(prompt)
            
            # Try to parse JSON response
            try:
                # Extract JSON from response text
                text = response.text
                # Find JSON block
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    print(f"    ✅ Gemini analysis complete")
                    return analysis
                else:
                    # Return raw text if JSON parsing fails
                    return {'raw_analysis': text}
            except Exception as e:
                print(f"    ⚠️ Could not parse Gemini response as JSON: {e}")
                return {'raw_analysis': response.text}
                
        except Exception as e:
            print(f"    ⚠️ Gemini analysis error: {e}")
            return {}
    
    async def search_with_gemini(self, company: str, query_type: str = 'hr_contacts') -> List[Dict]:
        """Use Gemini with Google Search to find HR contacts"""
        if not GEMINI_API_KEY:
            print("    ⚠️ Gemini API key not found, skipping Gemini search")
            return []
        
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=GEMINI_API_KEY)
            
            # Strict prompt for deep search of Talent Acquisition & Technical Recruiting HRs
            if query_type == 'hr_contacts':
                prompt = f"""Conduct a DEEP SEARCH for HR personnel at {company}. 

STRICT SEARCH FOCUS:
1. Talent Acquisition Specialists / タレントアクイジション担当
2. Technical Recruiters / テクニカルリクルーター
3. HR Managers / HRマネージャー
4. Hiring Managers / 採用担当
5. Recruitment Leads / リクルートメントリード
6. People Operations / ピープルオペレーション

REQUIRED OUTPUT FORMAT:
For each person found, return EXACTLY this format:
Name: [Full Name]
Position: [Job Title/Role]
Email: [email if found, otherwise "Not found"]
LinkedIn: [LinkedIn URL if found, otherwise "Not found"]
Source: [Where you found this information]

SEARCH INSTRUCTIONS:
- Search LinkedIn profiles with site:linkedin.com/in
- Search company career pages
- Search recruitment platforms (Wantedly, Indeed, Daijob, Mynavi, Tenshoku Mynavi)
- Look for press releases about HR appointments
- Search for email patterns (@company.com, @paypay.ne.jp, etc.)

Return as many actual HR contacts as possible with complete information."""
            elif query_type == 'emails':
                prompt = f"""Conduct a DEEP SEARCH for HR email addresses at {company}.

TARGET ROLES:
- Talent Acquisition / タレントアクイジション
- Technical Recruiting / テクニカルリクルーティング
- HR Department / 人事部
- Recruitment Team / 採用チーム

SEARCH STRATEGY:
1. Search for "@{company.split()[0].lower()}.com" or "@{company.split()[0].lower()}.co.jp" patterns
2. Search LinkedIn for HR profiles with visible contact info
3. Search email finder sites (RocketReach, Hunter.io, ContactOut, Mynavi)
4. Look for press releases with contact information

REQUIRED OUTPUT FORMAT:
Name: [Person Name]
Position: [Role]
Email: [email address]
Confidence: [High/Medium/Low]
Source: [URL or source]

Return all valid email addresses found."""
            else:
                prompt = f"""Search for comprehensive HR department information at {company}.

Include:
- HR department structure
- Key HR personnel names and titles
- Contact information
- LinkedIn company page
- Career/recruitment page URLs

Format as structured data with clear labels."""
            
            print(f"    🔍 Gemini DEEP SEARCH: {query_type} for {company}")
            
            # Use Gemini with Google Search tool enabled
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )
            
            # Extract information from response
            results = []
            text = response.text
            
            # Parse structured format: Name: ... Position: ... Email: ...
            import re
            
            # Look for structured entries
            entries = re.split(r'\n\n+', text)
            for entry in entries:
                name_match = re.search(r'Name:\s*([^\n]+)', entry, re.IGNORECASE)
                position_match = re.search(r'Position:\s*([^\n]+)', entry, re.IGNORECASE)
                email_match = re.search(r'Email:\s*([^\n\s]+)', entry, re.IGNORECASE)
                linkedin_match = re.search(r'LinkedIn:\s*([^\n\s]+)', entry, re.IGNORECASE)
                
                if name_match:
                    name = name_match.group(1).strip()
                    if name and name.lower() != 'not found':
                        results.append({
                            'type': 'name',
                            'value': name,
                            'position': position_match.group(1).strip() if position_match else '',
                            'source': 'gemini_search'
                        })
                
                if email_match:
                    email = email_match.group(1).strip()
                    if email and '@' in email and 'not found' not in email.lower():
                        results.append({
                            'type': 'email',
                            'value': email,
                            'source': 'gemini_search'
                        })
                
                if linkedin_match:
                    linkedin = linkedin_match.group(1).strip()
                    if linkedin and 'linkedin.com' in linkedin and 'not found' not in linkedin.lower():
                        results.append({
                            'type': 'url',
                            'value': linkedin,
                            'source': 'gemini_search'
                        })
            
            # Also use regex extraction as fallback
            names = self.extract_names_with_llm(text, company)
            emails = self.extract_emails(text)
            
            for name_info in names:
                if not any(r['type'] == 'name' and r['value'] == name_info['name'] for r in results):
                    results.append({
                        'type': 'name',
                        'value': name_info['name'],
                        'source': 'gemini_search'
                    })
            
            for email in emails:
                if not any(r['type'] == 'email' and r['value'] == email for r in results):
                    results.append({
                        'type': 'email',
                        'value': email,
                        'source': 'gemini_search'
                    })
            
            name_count = len([r for r in results if r['type'] == 'name'])
            email_count = len([r for r in results if r['type'] == 'email'])
            url_count = len([r for r in results if r['type'] == 'url'])
            
            print(f"    ✅ Gemini DEEP SEARCH found: {name_count} names, {email_count} emails, {url_count} LinkedIn URLs")
            return results
            
        except Exception as e:
            print(f"    ⚠️ Gemini search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def extract_hr_with_ollama(self, raw_text: str, company: str, search_query: str) -> Dict:
        """Use Ollama local LLM to extract HR contacts from raw search results"""
        try:
            import aiohttp
            
            # Prepare prompt for Ollama
            prompt = f"""You are an expert HR contact extractor. Analyze the following search results for {company} and extract HR personnel information.

Search Query: {search_query}

Raw Search Results:
{raw_text[:8000]}  # Limit text to avoid token limits

Instructions:
1. Extract ALL person names that appear to be HR personnel, recruiters, or talent acquisition specialists
2. Extract ALL email addresses
3. Extract ALL LinkedIn profile URLs
4. For each person, determine their role/title if available
5. Return results in strict JSON format

Return JSON format:
{{
    "people": [
        {{
            "name": "Full Name",
            "title": "Job Title",
            "email": "email@example.com or null",
            "linkedin": "linkedin url or null",
            "confidence": "high/medium/low"
        }}
    ],
    "emails": ["email1@example.com", "email2@example.com"],
    "urls": ["https://..."],
    "notes": "Any additional insights"
}}

If no HR contacts found, return empty arrays."""

            # Call Ollama API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 2000
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get('response', '')
                        
                        # Try to extract JSON from response
                        try:
                            # Find JSON block
                            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                            if json_match:
                                data = json.loads(json_match.group())
                                print(f"      🤖 Ollama found {len(data.get('people', []))} people, {len(data.get('emails', []))} emails")
                                return data
                        except Exception as e:
                            print(f"      ⚠️ Could not parse Ollama response: {e}")
                            return {'raw_response': response_text}
                    else:
                        print(f"      ⚠️ Ollama API error: {response.status}")
                        return {}
                        
        except Exception as e:
            print(f"      ⚠️ Ollama extraction error: {e}")
            return {}
    
    def extract_emails(self, text: str) -> List[str]:
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = list(set(re.findall(pattern, text)))
        
        filtered = []
        for email in emails:
            email_lower = email.lower()
            # Filter out fake/example emails
            if any(x in email_lower for x in [
                'example.com', 'test.com', '.png', '.jpg', 'w3.org',
                'sentry.io', 'yourdomain.com', 'email.com'
            ]):
                continue
            # Filter out support/customer service emails (not HR)
            if any(x in email_lower for x in [
                'support@', 'support2@', 'support3@', 'help@', 'info@', 
                'contact@', 'hello@', 'service@', 'cs@', 'customer@',
                'feedback@', 'inquiry@', 'questions@'
            ]):
                continue
            filtered.append(email)
        return filtered
    
    async def crawl_page(self, url: str) -> str:
        """Crawl a single page"""
        try:
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.DISABLED,
                page_timeout=15000
            )
            
            result = await self.crawler.arun(url=url, config=run_config)
            
            if result.success:
                return result.markdown or result.text or ""
        except RuntimeError as e:
            # Crawl4AI/Playwright timeout or navigation errors
            if "Timeout" in str(e) or "net::" in str(e):
                print(f"      ⏱️ Timeout/Network error for {url[:50]}...")
            else:
                print(f"      ⚠️ Runtime error crawling {url[:50]}: {str(e)[:100]}")
        except Exception as e:
            print(f"      ⚠️ Error crawling {url[:50]}: {str(e)[:100]}")
        
        return ""
    
    async def search_google(self, query: str) -> Dict:
        """Search using Google with Bright Data Scraping Browser (Selenium)"""
        from urllib.parse import quote_plus
        
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        print(f"    Google Search: {query[:50]}...")
        
        # Bright Data Scraping Browser credentials (Selenium on port 9515)
        AUTH = 'brd-customer-hl_f91d7a35-zone-web_unlocker1:1ol5mjpwv83b'
        
        try:
            from selenium.webdriver import Remote, ChromeOptions as Options
            from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection as Connection
            
            server_addr = f'https://{AUTH}@brd.superproxy.io:9515'
            connection = Connection(server_addr, 'goog', 'chrome')
            driver = Remote(connection, options=Options())
            
            try:
                # Navigate to Google search
                driver.get(search_url)
                
                # Get page source
                html = driver.page_source
                print(f"      Page loaded, HTML length: {len(html)}")
                
                # Parse with BeautifulSoup
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                # Get text from HTML for email extraction
                text = soup.get_text(separator=' ', strip=True)
                emails = self.extract_emails(text)
                
                # Parse search results (Google format)
                search_results = []
                
                # Google uses various selectors for results
                for g in soup.find_all('div', class_=lambda x: x and ('g' == x or 'yuRUbf' in x or 'v7W49e' in x if x else False)):
                    title_elem = g.find('h3')
                    link_elem = g.find('a')
                    if title_elem and link_elem:
                        search_results.append({
                            'title': title_elem.get_text(),
                            'url': link_elem.get('href', '')
                        })
                
                # Also try generic patterns
                if not search_results:
                    for g in soup.find_all(['div', 'article'], class_=lambda x: x and ('result' in x.lower() if x else False)):
                        title_elem = g.find('h3') or g.find('a')
                        link_elem = g.find('a')
                        if title_elem and link_elem:
                            search_results.append({
                                'title': title_elem.get_text(),
                                'url': link_elem.get('href', '')
                            })
                
                print(f"      Found {len(search_results)} results, {len(emails)} emails")
                if search_results:
                    print(f"      First result: {search_results[0]['title'][:50]}...")
                
                # Use Ollama LLM to extract HR data from raw text
                print(f"      🤖 Sending to Ollama LLM for analysis...")
                ollama_data = await self.extract_hr_with_ollama(text, query.split()[0], query)
                
                return {
                    'status': 'success',
                    'results': search_results[:10],
                    'emails': emails,
                    'text': text,
                    'ollama_data': ollama_data  # Add Ollama extracted data
                }
            finally:
                driver.quit()
                
        except Exception as e:
            print(f"      Error with Selenium: {e}")
            return {'status': 'error', 'results': [], 'emails': [], 'ollama_data': {}}
    
    def extract_person_names(self, text: str, company: str) -> List[Dict]:
        """Extract person names from text - enhanced for LinkedIn-style results"""
        names = []
        
        # Pattern 1: LinkedIn style - "First Last - Title at Company"
        linkedin_patterns = [
            r'([A-Z][a-z]+\s[A-Z][a-z]+)\s*[-–—]\s*(?:Tech\s+)?(?:Recruiter|HR|Talent|採用|人事).*?(?:at|@).*?(?:corporate|株式会社)',
            r'LinkedIn\s*[-–—]\s*([A-Z][a-z]+\s[A-Z][a-z]+)',
            r'([A-Z][a-z]+\s[A-Z][a-z]+)\s*[-–—]\s*(?:HR|人事|Recruiter|採用|Talent)',
        ]
        
        # Pattern 2: English names with titles
        title_patterns = [
            r'([A-Z][a-z]+\s[A-Z][a-z]+)\s*[-–—]\s*(?:HR|人事|Recruiter|採用|Talent|Manager)',
            r'(?:HR|人事|Recruiter|採用|Talent)\s*[-–—]\s*([A-Z][a-z]+\s[A-Z][a-z]+)',
        ]
        
        # Pattern 3: Japanese names (2-4 kanji characters)
        japanese_pattern = r'[\u4E00-\u9FAF]{2,4}(?:\s+[\u4E00-\u9FAF]{2,4})?'
        
        # Try LinkedIn patterns first
        for pattern in linkedin_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1).strip()
                if len(name) > 3 and not any(x in name.lower() for x in ['inc', 'corp', '株式会社']):
                    names.append({'name': name, 'type': 'linkedin'})
        
        # Try title patterns
        for pattern in title_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if len(name) > 3:
                    names.append({'name': name, 'type': 'english'})
        
        # Find Japanese names - filter out common non-name words
        common_japanese_words = {
            '株式会社', '会社概要', '採用情報', 'お問い合わせ', 'ホームページ',
            '採用担当', '人事部', '人事担当', '採用課', '人事課', '採用係',
            '採用チーム', '人事チーム', '採用部', '採用課長', '人事課長',
            '採用部長', '人事部長', '採用次長', '人事次長', '採用担当者',
            '人事担当者', '採用責任者', '人事責任者', '採用窓口', '人事窓口',
            '採用連絡', '人事連絡', '採用問合', '人事問合', '採用相談',
            '人事相談', '採用応募', '人事応募', '採用選考', '人事選考',
            '採用面接', '人事面接', '採用書類', '人事書類', '採用案内',
            '人事案内', '採用情報', '人事情報', '採用募集', '人事募集',
            '採用要項', '人事要項', '採用規定', '人事規定', '採用条件',
            '人事条件', '採用待遇', '人事待遇', '採用福利', '人事福利',
            '採用休暇', '人事休暇', '採用保険', '人事保険', '採用年金',
            '人事年金', '採用給与', '人事給与', '採用手当', '人事手当',
            '採用賞与', '人事賞与', '採用昇給', '人事昇給', '採用評価',
            '人事評価', '採用教育', '人事教育', '採用研修', '人事研修',
            '採用訓練', '人事訓練', '採用育成', '人事育成', '採用開発',
            '人事開発', '採用配置', '人事配置', '採用異動', '人事異動',
            '採用転勤', '人事転勤', '採用出向', '人事出向', '採用休職',
            '人事休職', '採用復職', '人事復職', '採用退職', '人事退職',
            '採用解雇', '人事解雇', '採用雇止', '人事雇止', '採用再雇用',
            '人事再雇用', '採用契約', '人事契約', '採用更新', '人事更新',
            '採用更改', '人事更改', '採用変更', '人事変更', '採用解除',
            '人事解除', '採用終了', '人事終了', '採用満了', '人事満了',
            '採用期間', '人事期間', '採用試用', '人事試用', '採用見習',
            '人事見習', '採用実習', '人事実習', '採用研修生', '人事研修生',
            '採用新入', '人事新入', '採用中途', '人事中途', '採用経験',
            '人事経験', '採用実績', '人事実績', '採用実務', '人事実務',
            '採用業務', '人事業務', '採用作業', '人事作業', '採用職務',
            '人事職務', '採用職場', '人事職場', '採用勤務', '人事勤務',
            '採用勤務地', '人事勤務地', '採用勤務時間', '人事勤務時間',
            '採用就業', '人事就業', '採用就労', '人事就労', '採用労働',
            '人事労働', '採用労務', '人事労務', '採用安全', '人事安全',
            '採用衛生', '人事衛生', '採用健康', '人事健康', '採用管理',
            '人事管理', '採用統括', '人事統括', '採用責任', '人事責任',
            '採用主管', '人事主管', '採用主査', '人事主査', '採用主幹',
            '人事主幹', '採用主任', '人事主任', '採用係長', '人事係長',
            '採用課長代理', '人事課長代理', '採用部長代理', '人事部長代理',
            '採用係長代理', '人事係長代理', '採用担当者', '人事担当者',
            '採用責任者', '人事責任者', '採用窓口', '人事窓口', '企業情報',
            '会社情報', '企業概要', '会社概要', '企業理念', '会社理念',
            '企業方針', '会社方針', '企業沿革', '会社沿革', '企業歴史',
            '会社歴史', '企業紹介', '会社紹介', '企業案内', '会社案内',
        }
        
        for match in re.finditer(japanese_pattern, text):
            name = match.group().strip()
            if 2 <= len(name) <= 5 and name not in common_japanese_words:
                names.append({'name': name, 'type': 'japanese'})
        
        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for n in names:
            if n['name'] not in seen:
                seen.add(n['name'])
                unique_names.append(n)
        
        return unique_names[:10]  # Return top 10
    
    def extract_names_with_llm(self, text: str, company: str) -> List[Dict]:
        """Use LLM to extract actual HR person names from search results"""
        names = []
        
        # Simple heuristic-based extraction (simulating LLM logic)
        # Look for patterns like "Name - Title at Company" or "Name | LinkedIn"
        
        # Pattern 1: English names (First Last) near HR keywords
        english_name_patterns = [
            r'([A-Z][a-z]+\s[A-Z][a-z]+)\s*(?:[-–|—])\s*(?:HR|Recruiter|Talent|採用|人事)',
            r'([A-Z][a-z]+\s[A-Z][a-z]+)\s*(?:at|@)\s*(?:Corp|株式会社)',
            r'(?:HR|Recruiter|Talent)\s*(?:[-–|—])\s*([A-Z][a-z]+\s[A-Z][a-z]+)',
            r'LinkedIn.*?(?:[-–|—])\s*([A-Z][a-z]+\s[A-Z][a-z]+)',
            r'([A-Z][a-z]+\s[A-Z][a-z]+)\s*[-–|—]\s*LinkedIn',
        ]
        
        for pattern in english_name_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1).strip()
                # Validate: should be 1-3 words, each capitalized, no common words
                words = name.split()
                if 1 <= len(words) <= 3 and all(w[0].isupper() for w in words if w):
                    # Exclude common false positives
                    common_words = ['Corp', 'Inc', 'Ltd', '株式会社', 'Company', 'Careers', 'Jobs', 'Recruiting', 'Hiring', 'Extension', 'Automation', 'Powered']
                    if not any(cw.lower() in name.lower() for cw in common_words):
                        if len(name) > 3:  # Minimum length for a real name
                            names.append({'name': name, 'type': 'english_llm'})
        
        # Pattern 1b: Japanese names with space separator (e.g., "坂本 良")
        japanese_name_with_space_pattern = r'([\u4E00-\u9FAF]{1,4})\s+([\u4E00-\u9FAF]{1,4})\s*(?:[-–—])\s*(?:HR|人事|採用|Recruiter|Talent|Manager)'
        for match in re.finditer(japanese_name_with_space_pattern, text):
            last_name = match.group(1)
            first_name = match.group(2)
            full_name = f"{last_name} {first_name}"
            names.append({'name': full_name, 'type': 'japanese_hr'})
        
        # Pattern 1c: Japanese names in format "Last First - Title" (no space between)
        japanese_name_compact_pattern = r'([\u4E00-\u9FAF]{2,4})\s*(?:[-–—])\s*(?:HR|人事|採用|Recruiter|Talent)'
        for match in re.finditer(japanese_name_compact_pattern, text):
            name = match.group(1)
            # Filter out common non-name words
            common_jp_words = {'人事', '採用', '担当', '責任者', '窓口', '連絡', '問合', '相談', '応募', '選考', '面接'}
            if name not in common_jp_words and len(name) >= 2:
                names.append({'name': name, 'type': 'japanese_compact'})
        
        # Pattern 2: Look for names in LinkedIn URLs
        linkedin_url_pattern = r'linkedin\.com/in/([a-z-]+)'
        for match in re.finditer(linkedin_url_pattern, text, re.IGNORECASE):
            url_name = match.group(1).replace('-', ' ').title()
            # Convert URL slug to potential name
            words = url_name.split()
            if 1 <= len(words) <= 3:
                # Filter out common non-name URL slugs
                non_names = ['company', 'school', 'jobs', 'careers', 'about', 'products', 'services']
                if not any(nn in url_name.lower() for nn in non_names):
                    names.append({'name': url_name, 'type': 'linkedin_url'})
        
        # Pattern 3: Japanese names with context
        # Look for patterns like "Name さん" or "Name 氏" or "Name - 職位"
        japanese_context_patterns = [
            r'([\u4E00-\u9FAF]{2,4})\s*(?:さん|氏|様)\s*(?:[-–|—])\s*(?:HR|人事|採用)',
            r'([\u4E00-\u9FAF]{2,4})\s*(?:[-–|—])\s*(?:HR|人事|採用|Recruiter)',
            r'(?:HR|人事|採用|Recruiter)\s*(?:[-–|—])\s*([\u4E00-\u9FAF]{2,4})',
            # Additional patterns for Japanese names
            r'([\u4E00-\u9FAF]{2,4})\s*[-–]\s*(?:Talent|Acquisition|Manager)',
            r'([\u4E00-\u9FAF]{2,4}[\s·•]+[\u4E00-\u9FAF]{2,4})',  # Name with separator
        ]
        
        common_japanese_words = {
            '株式会社', '会社概要', '採用情報', 'お問い合わせ', 'ホームページ',
            '採用担当', '人事部', '人事担当', '採用課', '人事課', '採用係',
            '事業成長', '変革', '証券株式', '会社', '日本', '企業', '情報',
            '採用', '人事', '担当', '責任者', '窓口', '連絡', '問合', '相談',
            '応募', '選考', '面接', '書類', '案内', '募集', '要項', '規定',
            '条件', '待遇', '福利', '休暇', '保険', '年金', '給与', '手当',
            '賞与', '昇給', '評価', '教育', '研修', '訓練', '育成', '開発',
            '配置', '異動', '転勤', '出向', '休職', '復職', '退職', '解雇',
            '雇止', '再雇用', '契約', '更新', '更改', '変更', '解除', '終了',
            '満了', '期間', '試用', '見習', '実習', '研修生', '新入', '中途',
            '経験', '実績', '実務', '業務', '作業', '職務', '職場', '勤務',
            '勤務地', '勤務時間', '就業', '就労', '労働', '労務', '安全', '衛生',
            '健康', '管理', '統括', '責任', '主管', '主査', '主幹', '主任',
            '係長', '課長代理', '部長代理', '係長代理', '担当者', '責任者',
            '企業情報', '会社情報', '企業概要', '会社概要', '企業理念', '会社理念',
            '企業方針', '会社方針', '企業沿革', '会社沿革', '企業歴史', '会社歴史',
            '企業紹介', '会社紹介', '企業案内', '会社案内', 'キャリア', '登録',
            '成長', '金融', '世界', '変更', '異動', 'お知らせ', 'プレス', 'リリース',
        }
        
        for pattern in japanese_context_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if name not in common_japanese_words and 2 <= len(name) <= 4:
                    names.append({'name': name, 'type': 'japanese_llm'})
        
        # Remove duplicates
        seen = set()
        unique_names = []
        for n in names:
            if n['name'] not in seen:
                seen.add(n['name'])
                unique_names.append(n)
        
        return unique_names
    
    async def find_hr_from_google(self, company: str) -> List[HRContact]:
        """Find HR people and emails using Google search with multiple keywords"""
        print(f"\n🔍 Google Search for HR at: {company}")
        
        contacts = []
        found_urls = set()
        found_names = set()
        
        # Multiple search queries for HR people
        people_queries = [
            f'{company} HR',
            f'{company} 人事',
            f'{company} HR manager',
            f'{company} HR 担当',
            f'{company} recruiter',
            f'{company} 採用担当',
            f'{company} talent acquisition',
            f'{company} HR person',
            f'{company} 人事部',
            f'{company} HR team',
            # Website-specific searches
            f'{company} linkedin HR',
            f'{company} linkedin 人事',
            f'{company} linkedin recruiter',
            f'{company} wantedly',
            f'{company} indeed',
            f'{company} contactout',
            f'{company} mynavi',
            f'{company} tenshoku.mynavi',
        ]
        
        # Multiple search queries for HR emails
        email_queries = [
            f'{company} HR email',
            f'{company} 人事 メール',
            f'{company} HR contact',
            f'{company} 採用 問い合わせ',
            f'{company} HR inquiry',
            f'{company} career email',
            f'{company} 採用メール',
            # Website-specific email searches
            f'{company} linkedin email',
            f'{company} contactout email',
            f'{company} wantedly contact',
            f'{company} mynavi email',
            f'{company} tenshoku.mynavi.jp',
        ]
        
        # Person name + keyword queries (will be populated when names are found)
        person_name_queries = []
        
        all_queries = people_queries + email_queries
        
        print(f"  📝 Running {len(all_queries)} search queries...")
        
        for query in all_queries[:6]:  # Limit to 6 queries to avoid rate limiting
            result = await self.search_google(query)
            
            if result['status'] == 'success' and result['results']:
                print(f"\n  📄 Query '{query[:40]}...' - {len(result['results'])} results")
                
                # Extract from full Google search page text (not just titles)
                full_text = result.get('text', '')
                all_names = self.extract_names_with_llm(full_text, company)
                
                if all_names:
                    print(f"    🔍 Found {len(all_names)} potential names in search results page")
                    for name_info in all_names:
                        name = name_info['name']
                        if name not in found_names:
                            found_names.add(name)
                            print(f"    👤 Found HR person: {name} ({name_info['type']})")
                            self.save_name_for_later(company, name, f"Google: {query}")
                
                # Also extract from individual result titles
                for item in result['results'][:5]:
                    title = item.get('title', '')
                    url = item.get('url', '')
                    
                    # Combine title and URL for better extraction
                    combined_text = f"{title} {url}"
                    
                    # Use LLM-based extraction
                    names = self.extract_names_with_llm(combined_text, company)
                    
                    if names:
                        print(f"    🔍 Found {len(names)} potential names in: {title[:50]}...")
                    
                    for name_info in names:
                        name = name_info['name']
                        if name not in found_names:
                            found_names.add(name)
                            print(f"    👤 Found HR person: {name} ({name_info['type']})")
                            
                            # Save name to names.json for later deep search
                            self.save_name_for_later(company, name, f"Google: {query}")
                            
                            contact = HRContact(
                                company=company,
                                company_url=item.get('url', ''),
                                hr_name=name,
                                title="HR Professional",
                                email="",
                                email_type="personal",
                                source=f"Google: {query}",
                                confidence=0.8
                            )
                            contacts.append(contact)
                            self.save_contact_immediately(contact, company)
                            
                            # Add person name + keyword queries for later searching
                            person_name_queries.extend([
                                f'{name} {company} email',
                                f'{name} {company} contact',
                                f'{name} {company} HR',
                                f'{name} {company} linkedin',
                                f'{name} {company} wantedly',
                                f'{name} {company} indeed',
                                f'{name} {company} contactout',
                                f'{name} {company} mynavi',
                                f'{name} {company} tenshoku.mynavi',
                            ])
                            
                            # Follow-up search: name + company + keywords (Google only, no URL crawling)
                            await self.search_person_details(name, company, found_names, contacts)
                
                # Note: We only extract from Google search results page, not from external URLs
                print(f"    ✓ Extracted from Google search results.")
        
        # Search for person name + keyword queries
        if person_name_queries:
            print(f"\n  🔍 Searching {len(person_name_queries)} person name + keyword queries...")
            for pn_query in person_name_queries[:6]:  # Limit to 6
                try:
                    result = await self.search_google(pn_query)
                    if result['status'] == 'success' and result['results']:
                        print(f"    📄 Person query '{pn_query[:40]}...' - {len(result['results'])} results")
                        
                        # Extract emails from Google search page
                        text_content = result.get('text', '')
                        emails = self.extract_emails(text_content)
                        
                        for email in emails:
                            print(f"    📧 Found email: {email}")
                            # Extract name from query
                            name_from_query = pn_query.split(company)[0].strip()
                            
                            # Save email to emails.json
                            self.save_email_for_later(company, email, f"Person search: {pn_query}", result['results'][0].get('url', '') if result['results'] else '')
                            
                            contact = HRContact(
                                company=company,
                                company_url=result['results'][0].get('url', '') if result['results'] else '',
                                hr_name=name_from_query,
                                title="HR Professional",
                                email=email,
                                email_type="personal",
                                source=f"Person search: {pn_query}",
                                confidence=0.85
                            )
                            contacts.append(contact)
                            self.save_contact_immediately(contact, company)
                except Exception as e:
                    print(f"    Error in person query: {e}")
        
        # Remove duplicates
        seen_emails = set()
        seen_names = set()
        unique_contacts = []
    
    async def search_person_details(self, person_name: str, company: str, found_names: set, contacts: list):
        """Search for specific person details on email finder sites"""
        print(f"    🔍 Searching details for: {person_name} at {company}")
        
        # Phase 1: Search email finder sites (Google search, crawl Google page)
        email_finder_queries = [
            f'{person_name} {company} wantedly email',
            f'{person_name} {company} indeed email',
            f'{person_name} {company} contactout email',
            f'{person_name} {company} rocketreach email',
            f'{person_name} {company} hunter.io email',
            f'{person_name} {company} voilanorbert email',
            f'{person_name} {company} skrapp email',
            f'{person_name} {company} apollo email',
            f'{person_name} {company} lusha email',
            f'{person_name} {company} snov email',
            f'{person_name} {company} anymail finder',
            f'{person_name} {company} email format',
            f'{person_name} {company} mynavi email',
            f'{person_name} {company} tenshoku.mynavi.jp',
        ]
        
        found_emails = []
        
        for query in email_finder_queries[:6]:  # Limit to 6 queries per person
            try:
                result = await self.search_google(query)
                
                if result['status'] == 'success' and result['results']:
                    print(f"      📄 Search: '{query[:45]}...' - {len(result['results'])} results")
                    
                    # Extract emails from Google search page only (no external URL crawling)
                    text_content = result.get('text', '')
                    emails = self.extract_emails(text_content)
                    
                    for email in emails:
                        if email not in found_emails:
                            found_emails.append(email)
                            print(f"      📧 Found email in Google results: {email}")
                            
                            # Save email to emails.json
                            self.save_email_for_later(company, email, f"Google search: {query}", result['results'][0].get('url', '') if result['results'] else '')
                            
                            contact = HRContact(
                                company=company,
                                company_url=result['results'][0].get('url', '') if result['results'] else '',
                                hr_name=person_name,
                                title="HR Professional",
                                email=email,
                                email_type="personal",
                                source=f"Google search: {query}",
                                confidence=0.85
                            )
                            contacts.append(contact)
                            self.save_contact_immediately(contact, company)
                    
                    # If we found emails, stop searching
                    if emails:
                        break
                            
            except Exception as e:
                print(f"      Error in search: {e}")
        
        # Note: We do NOT crawl external URLs - only extract from Google search results page
        
    async def find_hr_from_google(self, company: str) -> List[HRContact]:
        """Find HR people and emails using Google search with multiple keywords"""
        print(f"\n🔍 Google Search for HR at: {company}")
            
        contacts = []
        found_urls = set()
        found_names = set()
            
        # Multiple search queries for HR people
        people_queries = [
            f'{company} HR',
            f'{company} 人事',
            f'{company} HR manager',
            f'{company} HR 担当',
            f'{company} recruiter',
            f'{company} 採用担当',
            f'{company} talent acquisition',
            f'{company} HR person',
            f'{company} 人事部',
            f'{company} HR team',
            # Website-specific searches
            f'{company} linkedin HR',
            f'{company} linkedin 人事',
            f'{company} linkedin recruiter',
            f'{company} wantedly',
            f'{company} indeed',
            f'{company} contactout',
        ]
            
        # Multiple search queries for HR emails
        email_queries = [
            f'{company} HR email',
            f'{company} 人事 メール',
            f'{company} HR contact',
            f'{company} 採用 問い合わせ',
            f'{company} HR inquiry',
            f'{company} career email',
            f'{company} 採用メール',
            # Website-specific email searches
            f'{company} linkedin email',
            f'{company} contactout email',
            f'{company} wantedly contact',
        ]
            
        all_queries = people_queries + email_queries
            
        print(f"  📝 Running {len(all_queries)} search queries...")
            
        for query in all_queries[:6]:  # Limit to 6 queries to avoid rate limiting
            result = await self.search_google(query)
                
            if result['status'] == 'success' and result['results']:
                print(f"\n  📄 Query '{query[:40]}...' - {len(result['results'])} results")
                
                # Process Ollama LLM extracted data
                ollama_data = result.get('ollama_data', {})
                print(f"    🔍 DEBUG: Ollama data type: {type(ollama_data)}, has people: {'people' in ollama_data}")
                if ollama_data and isinstance(ollama_data, dict):
                    people_list = ollama_data.get('people', [])
                    print(f"    🔍 DEBUG: Processing {len(people_list)} people from Ollama")
                    # Process people found by Ollama
                    for person in people_list:
                        name = person.get('name', '').strip()
                        if name and name not in found_names:
                            found_names.add(name)
                            print(f"    👤 Ollama found HR person: {name} ({person.get('title', 'Unknown')})")
                            self.save_name_for_later(company, name, f"Ollama LLM: {query}")
                            
                            contact = HRContact(
                                company=company,
                                company_url=person.get('linkedin', ''),
                                hr_name=name,
                                title=person.get('title', 'HR Professional'),
                                email=person.get('email', ''),
                                email_type="personal",
                                source=f"Ollama LLM: {query}",
                                confidence=0.9 if person.get('confidence') == 'high' else 0.7
                            )
                            contacts.append(contact)
                            self.save_contact_immediately(contact, company)
                            
                            # Follow-up search: name + company + keywords
                            await self.search_person_details(name, company, found_names, contacts)
                    
                    # Process emails found by Ollama
                    for email in ollama_data.get('emails', []):
                        if email and email not in [c.email for c in contacts]:
                            print(f"    📧 Ollama found email: {email}")
                            self.save_email_for_later(company, email, f"Ollama LLM: {query}", result['results'][0].get('url', '') if result['results'] else '')
                    
                    # Process URLs found by Ollama
                    for url in ollama_data.get('urls', []):
                        if url and url not in found_urls:
                            found_urls.add(url)
                            self.save_url_for_later(company, url, f"Ollama LLM: {query}")
                
                # Also use traditional regex extraction as fallback
                for item in result['results'][:5]:
                    title = item.get('title', '')
                    url = item.get('url', '')
                        
                    # Combine title and URL for better extraction
                    combined_text = f"{title} {url}"
                        
                    # Use LLM-based extraction
                    names = self.extract_names_with_llm(combined_text, company)
                        
                    for name_info in names:
                        name = name_info['name']
                        if name not in found_names:
                            found_names.add(name)
                            print(f"    👤 Found HR person: {name} ({name_info['type']})")
                            contacts.append(HRContact(
                                company=company,
                                company_url=item.get('url', ''),
                                hr_name=name,
                                title="HR Professional",
                                email="",
                                email_type="personal",
                                source=f"Google: {query}",
                                confidence=0.8
                            ))
                                
                            # Follow-up search: name + company + keywords
                            await self.search_person_details(name, company, found_names, contacts)
                    
                # Collect URLs from search results (for later crawling - NOT crawling now)
                for item in result['results'][:5]:
                    url = item.get('url', '')
                    if url and url not in found_urls:
                        if not any(x in url for x in ['google.com', 'google.co.jp']):
                            found_urls.add(url)
                            # Save URL for later crawling (not crawling now)
                            self.save_url_for_later(company, url, item.get('title', ''))
            
        # Remove duplicates
        seen_emails = set()
        seen_names = set()
        unique_contacts = []
            
        for c in contacts:
            if c.email and c.email not in seen_emails:
                seen_emails.add(c.email)
                unique_contacts.append(c)
            elif c.hr_name and c.hr_name not in seen_names and not c.email:
                seen_names.add(c.hr_name)
                unique_contacts.append(c)
            
        print(f"\n  ✅ Found {len(unique_contacts)} unique HR contacts ({len(seen_names)} names, {len(seen_emails)} emails)")
        return unique_contacts
    
    async def find_company_website(self, company: str) -> str:
        """Search for company official website using Google"""
        print(f"  🔍 Searching Google for {company} official website...")
        
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from urllib.parse import urlparse
            
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            # Try to use Bright Data if available
            brightdata_user = os.getenv('BRIGHTDATA_USER', '')
            brightdata_pass = os.getenv('BRIGHTDATA_PASS', '')
            
            if brightdata_user and brightdata_pass:
                # Use Bright Data Scraping Browser
                selenium_url = f"http://{brightdata_user}:{brightdata_pass}@brd.superproxy.io:9515"
                driver = webdriver.Remote(command_executor=selenium_url, options=chrome_options)
            else:
                # Use local Chrome
                driver = webdriver.Chrome(options=chrome_options)
            
            try:
                # Search for company website
                search_query = f"{company} official website"
                driver.get(f"https://www.google.com/search?q={search_query}")
                
                # Wait for results
                wait = WebDriverWait(driver, 10)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.g')))
                
                # Get first result
                results = driver.find_elements(By.CSS_SELECTOR, 'div.g')
                for result in results[:3]:
                    try:
                        link = result.find_element(By.CSS_SELECTOR, 'a')
                        url = link.get_attribute('href')
                        
                        # Skip Google, LinkedIn, and job sites
                        if url and not any(x in url for x in ['google.com', 'linkedin.com', 'indeed.com', 'glassdoor.com', 'wikipedia.org']):
                            # Check if URL looks like official company site
                            parsed = urlparse(url)
                            domain = parsed.netloc.lower()
                            
                            # Extract company name parts
                            company_lower = company.lower()
                            for suffix in ['株式会社', 'inc.', 'corp.', 'corporation', 'limited', 'ltd.', 'co.,', 'co.', 'kk', '合同会社']:
                                company_lower = company_lower.replace(suffix, '').strip()
                            
                            # Check if domain contains company name
                            company_parts = company_lower.split()
                            for part in company_parts:
                                if len(part) >= 3 and part in domain:
                                    print(f"  ✅ Found company website: {url}")
                                    return url
                            
                            # If no match but looks like a corporate site, return it
                            if '.co.jp' in domain or '.jp' in domain or '.com' in domain:
                                print(f"  ✅ Found potential website: {url}")
                                return url
                                
                    except Exception as e:
                        continue
                
                print(f"  ⚠️ Could not find official website for {company}")
                return ""
                
            finally:
                driver.quit()
                
        except Exception as e:
            print(f"  ⚠️ Error finding website: {e}")
            return ""
    
    async def find_company_hr(self, company: str, company_url: str) -> List[HRContact]:
        """Find HR contacts using 4 methods: Crawl4AI, Gemini Search, Google Search, LLM Processing"""
        print(f"\n{'='*80}")
        print(f"🔍 4-METHOD HR SEARCH: {company}")
        print(f"{'='*80}")
        print(f"Methods: Crawl4AI + Gemini Search + Google Search + Ollama LLM")
        print(f"{'='*80}")
        
        contacts = []
        found_urls = set()
        found_names = set()
        
        # ========== METHOD 1: CRAWL4AI - Direct Website Crawling ==========
        print(f"\n📌 METHOD 1: CRAWL4AI - Direct Website Crawling")
        
        # If no company URL provided, try to find it via Google search
        if not company_url or not str(company_url).startswith('http'):
            print(f"  🔍 No URL provided, searching for {company} official website...")
            company_url = await self.find_company_website(company)
        
        if company_url and str(company_url).startswith('http'):
            print(f"  🌐 Using company URL: {company_url}")
            pages = ['', '/careers', '/jobs', '/recruit', '/about', '/contact', '/company', '/about/careers']
            for page in pages:
                url = f"{company_url.rstrip('/')}{page}"
                print(f"  🔗 Crawling: {url}")
                text = await self.crawl_page(url)
                if text:
                    emails = self.extract_emails(text)
                    for email in emails:
                        print(f"    📧 Found email: {email}")
                        self.save_email_for_later(company, email, f"Crawl4AI: {url}", url)
                        contacts.append(HRContact(
                            company=company,
                            company_url=url,
                            hr_name="HR Team",
                            title="Human Resources",
                            email=email,
                            email_type="hr_department",
                            source=f"Crawl4AI: {page or 'main'}",
                            confidence=0.8
                        ))
                        self.save_contact_immediately(contacts[-1], company)
        else:
            print(f"  ⚠️ Could not find company website, skipping Crawl4AI")
        
        # ========== METHOD 2: GEMINI SEARCH - AI-Powered Search ==========
        print(f"\n📌 METHOD 2: GEMINI SEARCH - AI-Powered Google Search")
        gemini_results = await self.search_with_gemini(company, 'hr_contacts')
        for result in gemini_results:
            if result['type'] == 'name':
                name = result['value']
                if name not in found_names:
                    found_names.add(name)
                    print(f"    👤 Gemini found: {name}")
                    self.save_name_for_later(company, name, 'Gemini AI search')
                    contacts.append(HRContact(
                        company=company,
                        company_url='',
                        hr_name=name,
                        title="HR Professional",
                        email="",
                        email_type="personal",
                        source="Gemini AI search",
                        confidence=0.85
                    ))
                    self.save_contact_immediately(contacts[-1], company)
            elif result['type'] == 'email':
                email = result['value']
                print(f"    📧 Gemini found: {email}")
                self.save_email_for_later(company, email, 'Gemini AI search', company_url)
        
        # ========== METHOD 3: GOOGLE SEARCH - Selenium ==========
        print(f"\n📌 METHOD 3: GOOGLE SEARCH ")
        await asyncio.sleep(2)
        google_contacts = await self.find_hr_from_google(company)
        for contact in google_contacts:
            if contact.hr_name and contact.hr_name not in found_names:
                found_names.add(contact.hr_name)
                contacts.append(contact)
                self.save_contact_immediately(contact, company)
            elif contact.email:
                contacts.append(contact)
                self.save_contact_immediately(contact, company)
        print(f"  ✅ Google Search found {len(google_contacts)} contacts")
        
        # ========== METHOD 4: OLLAMA LLM - Process All Raw Data ==========
        print(f"\n📌 METHOD 4: OLLAMA LLM - Deep Analysis of All Data")
        # Load all collected data for analysis
        names_file = Path(".") / "names.json"
        all_names = []
        if names_file.exists():
            with open(names_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_names = [n['name'] for n in data.get(company, [])]
        
        emails_file = Path(".") / "emails.json"
        all_emails = []
        if emails_file.exists():
            with open(emails_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_emails = [e['email'] for e in data.get(company, [])]
        
        urls_file = Path(".") / "urls.json"
        all_urls = []
        if urls_file.exists():
            with open(urls_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_urls = [u['url'] for u in data.get(company, [])]
        
        print(f"  📊 Data collected: {len(all_names)} names, {len(all_emails)} emails, {len(all_urls)} URLs")
        
        # Get Gemini analysis (uses all 4 methods' data)
        analysis = await self.analyze_with_gemini(company, all_names, all_emails, all_urls)
        
        if analysis:
            # Save analysis to file
            analysis_file = Path(".") / f"gemini_analysis_{company.replace(' ', '_').replace('/', '_')}.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            print(f"    💾 Saved Gemini analysis to {analysis_file.name}")
            
            # Display key insights
            if 'likely_hr_names' in analysis:
                print(f"\n    🎯 Top HR Names (according to Gemini):")
                for name in analysis['likely_hr_names'][:5]:
                    print(f"       - {name}")
            
            if 'likely_hr_emails' in analysis:
                print(f"\n    📧 Top HR Emails (according to Gemini):")
                for email in analysis['likely_hr_emails'][:5]:
                    print(f"       - {email}")
            
            if 'search_suggestions' in analysis:
                print(f"\n    💡 Gemini Search Suggestions:")
                for suggestion in analysis['search_suggestions'][:3]:
                    print(f"       - {suggestion}")
        
        # Remove duplicates by email
        seen_emails = set()
        unique_contacts = []
        for c in contacts:
            if c.email and c.email not in seen_emails:
                seen_emails.add(c.email)
                unique_contacts.append(c)
            elif not c.email:
                unique_contacts.append(c)
        
        # Sort by confidence
        unique_contacts.sort(key=lambda x: x.confidence, reverse=True)
        return unique_contacts[:5]


def parse_args():
    """Parse command line arguments"""
    args = {
        'test': '--test' in sys.argv,
        'search': None
    }
    
    # Check for --search flag
    for i, arg in enumerate(sys.argv):
        if arg == '--search' and i + 1 < len(sys.argv):
            args['search'] = sys.argv[i + 1]
    
    return args


async def main():
    print("=" * 80)
    print("CRAWL4AI HR HUNTER")
    print("Google Search")
    print("=" * 80)
    
    args = parse_args()
    
    # Load data
    df = pd.read_excel("aitf様_企業名募集職種調査.xlsx")
    
    # Check if searching for specific company
    if args['search']:
        search_term = args['search']
        print(f"\n🔍 Searching for: {search_term}")
        
        # Filter companies that match search term
        companies_df = df[df['企業名（日本語正式名称）'].str.contains(search_term, case=False, na=False)]
        
        if companies_df.empty:
            print(f"❌ No company found matching '{search_term}'")
            print("\nAvailable companies:")
            for name in df['企業名（日本語正式名称）'].head(20):
                print(f"  - {name}")
            return
        
        companies_df = companies_df[['企業名（日本語正式名称）', '企業HP URL']].drop_duplicates()
        print(f"Found {len(companies_df)} matching company/companies")
    else:
        companies_df = df[['企業名（日本語正式名称）', '企業HP URL']].drop_duplicates()
        
        if args['test']:
            print(f"\n🧪 TEST MODE - Processing {TEST_JOB_COUNT} companies")
            companies_df = companies_df.head(TEST_JOB_COUNT)
    
    print(f"\nTotal companies to process: {len(companies_df)}")
    
    # Initialize hunter
    hunter = Crawl4AIDirectHunter()
    all_contacts = []
    
    try:
        await hunter.init_crawler()
        
        for _, row in companies_df.iterrows():
            company = row['企業名（日本語正式名称）']
            company_url = row['企業HP URL']
            
            contacts = await hunter.find_company_hr(company, company_url)
            all_contacts.extend(contacts)
    finally:
        await hunter.close()
    
    # LLM-based filtering and formatting
    all_contacts = await hunter.filter_and_format_results(all_contacts, "")
    
    # Statistics
    print("\n" + "=" * 80)
    print("FINAL FILTERED RESULTS")
    print("=" * 80)
    
    hr_contacts = [c for c in all_contacts if c.email_type == 'hr_department']
    general_contacts = [c for c in all_contacts if c.email_type == 'company_general']
    personal_contacts = [c for c in all_contacts if c.email_type == 'personal']
    
    print(f"Total contacts found: {len(all_contacts)}")
    print(f"Personal HR emails: {len(personal_contacts)}")
    print(f"HR department emails: {len(hr_contacts)}")
    print(f"General contacts: {len(general_contacts)}")
    
    # Show results - prioritize personal emails with names
    print("\n" + "=" * 80)
    print("TOP HR CONTACTS (Prioritized)")
    print("=" * 80)
    
    # Show personal emails first (with names)
    for contact in all_contacts[:30]:
        print(f"\n🏢 {contact.company}")
        print(f"   Name: {contact.hr_name}")
        print(f"   Email: {contact.email}")
        print(f"   Type: {contact.email_type}")
        print(f"   Source: {contact.source[:50]}...")
        print(f"   Confidence: {contact.confidence:.2f}")
    
    # Save final filtered results
    output_path = Path(".")
    
    json_data = [asdict(c) for c in all_contacts]
    with open(output_path / "crawl4ai_direct_results.json", 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    if all_contacts:
        df_out = pd.DataFrame([asdict(c) for c in all_contacts])
        df_out.to_csv(output_path / "crawl4ai_direct_summary.csv", index=False, encoding='utf-8-sig')
        print(f"\n✅ Saved {len(all_contacts)} filtered contacts")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
