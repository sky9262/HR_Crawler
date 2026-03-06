"""
HR Hunter Web Application
Upload Excel file, select column, and search for HR contacts
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import json
import os
import asyncio
from pathlib import Path
from datetime import datetime
import threading
import uuid

# Import the HR hunter class
from hr_hunter import Crawl4AIDirectHunter as HRHunter, HRContact

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store job status - use file-based storage for persistence across reloads
JOBS_FILE = Path('jobs.json')

def load_jobs():
    """Load jobs from file"""
    if JOBS_FILE.exists():
        try:
            with open(JOBS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_jobs(jobs):
    """Save jobs to file"""
    try:
        with open(JOBS_FILE, 'w') as f:
            json.dump(jobs, f)
    except Exception as e:
        print(f"Error saving jobs: {e}")

def get_jobs():
    """Get jobs dict (loads from file each time for persistence)"""
    return load_jobs()

def update_job(job_id, job_data):
    """Update a job and save to file"""
    jobs = load_jobs()
    jobs[job_id] = job_data
    save_jobs(jobs)

# Initialize jobs
jobs = get_jobs()


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload Excel file and return column names"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith(('.xlsx', '.xls')):
        return jsonify({'error': 'Please upload an Excel file (.xlsx or .xls)'}), 400
    
    try:
        # Save file
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read Excel and get columns
        df = pd.read_excel(filepath)
        columns = df.columns.tolist()
        
        # Get preview of first few rows (handle NaN values)
        preview_df = df.head(5).fillna('')
        preview = preview_df.to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'filename': file.filename,
            'columns': columns,
            'preview': preview,
            'total_rows': len(df)
        })
    
    except Exception as e:
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500


@app.route('/get_column_values', methods=['POST'])
def get_column_values():
    """Get unique values from a column"""
    data = request.json
    file_id = data.get('file_id')
    column = data.get('column')
    
    if not file_id or not column:
        return jsonify({'error': 'File ID and column required'}), 400
    
    try:
        # Find the file
        uploaded_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(file_id)]
        if not uploaded_files:
            return jsonify({'error': 'File not found'}), 404
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_files[0])
        df = pd.read_excel(filepath)
        
        # Get unique values (remove duplicates and NaN)
        values = df[column].dropna().unique().tolist()
        values = [str(v) for v in values if str(v).strip() and v != 'nan']
        
        return jsonify({
            'success': True,
            'values': values,
            'count': len(values)
        })
    
    except Exception as e:
        return jsonify({'error': f'Error reading column: {str(e)}'}), 500


@app.route('/start_search', methods=['POST'])
def start_search():
    """Start HR search for selected companies"""
    data = request.json
    file_id = data.get('file_id')
    column = data.get('column')
    companies = data.get('companies', [])
    
    if not file_id or not column or not companies:
        return jsonify({'error': 'File ID, column, and companies required'}), 400
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    job_data = {
        'id': job_id,
        'status': 'running',
        'total': len(companies),
        'completed': 0,
        'current': None,
        'results': [],
        'errors': [],
        'started_at': datetime.now().isoformat()
    }
    update_job(job_id, job_data)
    
    # Start search in background thread
    thread = threading.Thread(
        target=run_hr_search,
        args=(job_id, companies)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'message': f'Started search for {len(companies)} companies'
    })


def run_hr_search(job_id, companies):
    """Run HR search in background"""
    import traceback
    
    async def search_all():
        hunter = None
        try:
            print(f"[Job {job_id}] Starting search for {len(companies)} companies...")
            hunter = HRHunter()
            await hunter.init_crawler()
            print(f"[Job {job_id}] Crawler initialized")
            
            for i, company in enumerate(companies):
                # Load current job state
                job = load_jobs().get(job_id, {})
                job['current'] = company
                update_job(job_id, job)
                
                print(f"[Job {job_id}] Searching: {company} ({i+1}/{len(companies)})")
                
                try:
                    # Run search for this company
                    contacts = await hunter.find_company_hr(company, "")
                    print(f"[Job {job_id}] Found {len(contacts)} contacts for {company}")
                    
                    # Process results
                    company_results = {
                        'company': company,
                        'contacts': []
                    }
                    
                    for contact in contacts:
                        company_results['contacts'].append({
                            'name': contact.hr_name,
                            'title': contact.title,
                            'email': contact.email,
                            'linkedin': contact.company_url if 'linkedin.com' in contact.company_url else '',
                            'source': contact.source,
                            'confidence': contact.confidence
                        })
                    
                    # Load current job state and update
                    job = load_jobs().get(job_id, {})
                    job['results'].append(company_results)
                    job['completed'] = i + 1
                    update_job(job_id, job)
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"[Job {job_id}] Error searching {company}: {error_msg}")
                    print(traceback.format_exc())
                    
                    # Load current job state and update
                    job = load_jobs().get(job_id, {})
                    job['errors'].append({
                        'company': company,
                        'error': error_msg
                    })
                    job['completed'] = i + 1
                    update_job(job_id, job)
            
            # Mark as completed
            job = load_jobs().get(job_id, {})
            job['status'] = 'completed'
            update_job(job_id, job)
            print(f"[Job {job_id}] Search completed")
            
        except Exception as e:
            error_msg = str(e)
            print(f"[Job {job_id}] Fatal error: {error_msg}")
            print(traceback.format_exc())
            job = load_jobs().get(job_id, {})
            job['status'] = 'error'
            job['error_message'] = error_msg
            update_job(job_id, job)
        
        finally:
            if hunter:
                try:
                    await hunter.close()
                    print(f"[Job {job_id}] Browser closed")
                except Exception as e:
                    print(f"[Job {job_id}] Error closing browser: {e}")
    
    # Run async function
    try:
        asyncio.run(search_all())
    except Exception as e:
        print(f"[Job {job_id}] asyncio error: {e}")
        print(traceback.format_exc())
        job = load_jobs().get(job_id, {})
        job['status'] = 'error'
        job['error_message'] = str(e)
        update_job(job_id, job)


@app.route('/job_status/<job_id>')
def get_job_status(job_id):
    """Get job status and results"""
    jobs = load_jobs()
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    return jsonify({
        'id': job['id'],
        'status': job['status'],
        'total': job['total'],
        'completed': job['completed'],
        'current': job['current'],
        'progress_percent': round((job['completed'] / job['total']) * 100, 1) if job['total'] > 0 else 0,
        'results': job['results'],
        'errors': job['errors']
    })


@app.route('/download_results/<job_id>')
def download_results(job_id):
    """Download results as CSV"""
    jobs = load_jobs()
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    # Create DataFrame from results
    rows = []
    for result in job['results']:
        company = result['company']
        if result['contacts']:
            for contact in result['contacts']:
                rows.append({
                    'Company': company,
                    'HR Name': contact['name'],
                    'Title': contact['title'],
                    'Email': contact['email'],
                    'LinkedIn': contact['linkedin'],
                    'Source': contact['source'],
                    'Confidence': contact['confidence']
                })
        else:
            rows.append({
                'Company': company,
                'HR Name': '',
                'Title': '',
                'Email': '',
                'LinkedIn': '',
                'Source': 'No contacts found',
                'Confidence': ''
            })
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_path = f"results_{job_id}.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    return send_file(output_path, as_attachment=True, download_name='hr_contacts_results.csv')


if __name__ == '__main__':
    app.run(debug=True, port=5001)
