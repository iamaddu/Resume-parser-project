"""
SQLite Database Manager for HR System
Persistent storage for candidates, notes, and interviews
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

class DatabaseManager:
    """Manages SQLite database for HR system"""
    
    def __init__(self, db_path='hr_database.db'):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Candidates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                status TEXT DEFAULT 'Screening',
                match_score REAL,
                experience INTEGER,
                education TEXT,
                skills TEXT,
                job_title TEXT,
                resume_text TEXT,
                reasons_selected TEXT,
                reasons_rejected TEXT,
                location TEXT,
                current_company TEXT,
                notice_period TEXT,
                expected_salary TEXT,
                current_salary TEXT,
                linkedin_url TEXT,
                github_url TEXT,
                source TEXT DEFAULT 'Resume Upload',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Notes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER,
                note_type TEXT,
                note_text TEXT,
                added_by TEXT DEFAULT 'HR',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (candidate_id) REFERENCES candidates (id)
            )
        ''')
        
        # Interviews table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER,
                round TEXT,
                interviewer TEXT,
                result TEXT,
                rating INTEGER,
                feedback TEXT,
                interview_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (candidate_id) REFERENCES candidates (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ==================== CANDIDATES ====================
    
    def save_candidate(self, candidate_data):
        """Save or update candidate"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if candidate exists
        cursor.execute('SELECT id FROM candidates WHERE name = ?', (candidate_data['name'],))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing
            cursor.execute('''
                UPDATE candidates SET
                    email = ?, phone = ?, status = ?, match_score = ?,
                    experience = ?, education = ?, skills = ?, job_title = ?,
                    resume_text = ?, reasons_selected = ?, reasons_rejected = ?,
                    location = ?, current_company = ?, notice_period = ?,
                    expected_salary = ?, current_salary = ?, linkedin_url = ?,
                    github_url = ?, source = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (
                candidate_data.get('email', ''),
                candidate_data.get('phone', ''),
                candidate_data.get('status', 'Screening'),
                candidate_data.get('match_score', 0),
                candidate_data.get('experience', 0),
                candidate_data.get('education', ''),
                json.dumps(candidate_data.get('skills', [])),
                candidate_data.get('job_title', ''),
                candidate_data.get('resume_text', ''),
                json.dumps(candidate_data.get('reasons_selected', [])),
                json.dumps(candidate_data.get('reasons_rejected', [])),
                candidate_data.get('location', ''),
                candidate_data.get('current_company', ''),
                candidate_data.get('notice_period', ''),
                candidate_data.get('expected_salary', ''),
                candidate_data.get('current_salary', ''),
                candidate_data.get('linkedin_url', ''),
                candidate_data.get('github_url', ''),
                candidate_data.get('source', 'Resume Upload'),
                existing[0]
            ))
            candidate_id = existing[0]
        else:
            # Insert new
            cursor.execute('''
                INSERT INTO candidates (
                    name, email, phone, status, match_score, experience,
                    education, skills, job_title, resume_text,
                    reasons_selected, reasons_rejected, location,
                    current_company, notice_period, expected_salary,
                    current_salary, linkedin_url, github_url, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                candidate_data['name'],
                candidate_data.get('email', ''),
                candidate_data.get('phone', ''),
                candidate_data.get('status', 'Screening'),
                candidate_data.get('match_score', 0),
                candidate_data.get('experience', 0),
                candidate_data.get('education', ''),
                json.dumps(candidate_data.get('skills', [])),
                candidate_data.get('job_title', ''),
                candidate_data.get('resume_text', ''),
                json.dumps(candidate_data.get('reasons_selected', [])),
                json.dumps(candidate_data.get('reasons_rejected', [])),
                candidate_data.get('location', ''),
                candidate_data.get('current_company', ''),
                candidate_data.get('notice_period', ''),
                candidate_data.get('expected_salary', ''),
                candidate_data.get('current_salary', ''),
                candidate_data.get('linkedin_url', ''),
                candidate_data.get('github_url', ''),
                candidate_data.get('source', 'Resume Upload')
            ))
            candidate_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        return candidate_id
    
    def get_candidate(self, candidate_id=None, name=None):
        """Get candidate by ID or name"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if candidate_id:
            cursor.execute('SELECT * FROM candidates WHERE id = ?', (candidate_id,))
        elif name:
            cursor.execute('SELECT * FROM candidates WHERE name = ?', (name,))
        else:
            return None
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_candidate_dict(row)
        return None
    
    def get_all_candidates(self):
        """Get all candidates"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM candidates ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_candidate_dict(row) for row in rows]
    
    def update_candidate_status(self, candidate_id, new_status):
        """Update candidate status"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE candidates SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (new_status, candidate_id))
        conn.commit()
        conn.close()
    
    def delete_candidate(self, candidate_id):
        """Delete candidate and all related data"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Delete related notes and interviews
        cursor.execute('DELETE FROM notes WHERE candidate_id = ?', (candidate_id,))
        cursor.execute('DELETE FROM interviews WHERE candidate_id = ?', (candidate_id,))
        cursor.execute('DELETE FROM candidates WHERE id = ?', (candidate_id,))
        
        conn.commit()
        conn.close()
    
    def search_candidates(self, query):
        """Search candidates by name, skills, or status"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        search_pattern = f'%{query}%'
        cursor.execute('''
            SELECT * FROM candidates 
            WHERE name LIKE ? OR skills LIKE ? OR status LIKE ?
            ORDER BY created_at DESC
        ''', (search_pattern, search_pattern, search_pattern))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_candidate_dict(row) for row in rows]
    
    # ==================== NOTES ====================
    
    def add_note(self, candidate_id, note_text, note_type='general'):
        """Add note for candidate"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO notes (candidate_id, note_type, note_text)
            VALUES (?, ?, ?)
        ''', (candidate_id, note_type, note_text))
        
        note_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return note_id
    
    def get_notes(self, candidate_id):
        """Get all notes for candidate"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM notes WHERE candidate_id = ?
            ORDER BY created_at DESC
        ''', (candidate_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_note_dict(row) for row in rows]
    
    def delete_note(self, note_id):
        """Delete a note"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM notes WHERE id = ?', (note_id,))
        conn.commit()
        conn.close()
    
    def update_note(self, note_id, note_text):
        """Update a note"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE notes SET note_text = ? WHERE id = ?', (note_text, note_id))
        conn.commit()
        conn.close()
    
    # ==================== INTERVIEWS ====================
    
    def add_interview(self, candidate_id, interview_data):
        """Add interview record"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO interviews (
                candidate_id, round, interviewer, result, rating, feedback
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            candidate_id,
            interview_data.get('round', ''),
            interview_data.get('interviewer', ''),
            interview_data.get('result', ''),
            interview_data.get('rating', 0),
            interview_data.get('feedback', '')
        ))
        
        interview_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return interview_id
    
    def get_interviews(self, candidate_id):
        """Get all interviews for candidate"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM interviews WHERE candidate_id = ?
            ORDER BY interview_date DESC
        ''', (candidate_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_interview_dict(row) for row in rows]
    
    def get_all_interviews(self):
        """Get all interviews"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT i.*, c.name as candidate_name
            FROM interviews i
            JOIN candidates c ON i.candidate_id = c.id
            ORDER BY i.interview_date DESC
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_interview_dict(row, include_name=True) for row in rows]
    
    def delete_interview(self, interview_id):
        """Delete an interview"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM interviews WHERE id = ?', (interview_id,))
        conn.commit()
        conn.close()
    
    def update_interview(self, interview_id, interview_data):
        """Update an interview"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE interviews SET
                round = ?, interviewer = ?, result = ?, rating = ?, feedback = ?
            WHERE id = ?
        ''', (
            interview_data.get('round'),
            interview_data.get('interviewer'),
            interview_data.get('result'),
            interview_data.get('rating'),
            interview_data.get('feedback'),
            interview_id
        ))
        conn.commit()
        conn.close()
    
    # ==================== STATISTICS ====================
    
    def get_statistics(self):
        """Get hiring statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total counts
        cursor.execute('SELECT COUNT(*) FROM candidates')
        total_candidates = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM notes')
        total_notes = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM interviews')
        total_interviews = cursor.fetchone()[0]
        
        # This month
        cursor.execute('''
            SELECT COUNT(*) FROM candidates 
            WHERE strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now')
        ''')
        candidates_this_month = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) FROM interviews 
            WHERE strftime('%Y-%m', interview_date) = strftime('%Y-%m', 'now')
        ''')
        interviews_this_month = cursor.fetchone()[0]
        
        # Status breakdown
        cursor.execute('SELECT status, COUNT(*) FROM candidates GROUP BY status')
        status_rows = cursor.fetchall()
        status_breakdown = {row[0]: row[1] for row in status_rows}
        
        conn.close()
        
        return {
            'total_candidates': total_candidates,
            'total_notes': total_notes,
            'total_interviews': total_interviews,
            'candidates_this_month': candidates_this_month,
            'interviews_this_month': interviews_this_month,
            'status_breakdown': status_breakdown
        }
    
    # ==================== HELPER METHODS ====================
    
    def _row_to_candidate_dict(self, row):
        """Convert database row to candidate dictionary"""
        return {
            'id': row[0],
            'name': row[1],
            'email': row[2],
            'phone': row[3],
            'status': row[4],
            'match_score': row[5],
            'experience': row[6],
            'education': row[7],
            'skills': json.loads(row[8]) if row[8] else [],
            'job_title': row[9],
            'resume_text': row[10],
            'reasons_selected': json.loads(row[11]) if row[11] else [],
            'reasons_rejected': json.loads(row[12]) if row[12] else [],
            'location': row[13] if len(row) > 13 else '',
            'current_company': row[14] if len(row) > 14 else '',
            'notice_period': row[15] if len(row) > 15 else '',
            'expected_salary': row[16] if len(row) > 16 else '',
            'current_salary': row[17] if len(row) > 17 else '',
            'linkedin_url': row[18] if len(row) > 18 else '',
            'github_url': row[19] if len(row) > 19 else '',
            'source': row[20] if len(row) > 20 else 'Resume Upload',
            'created_at': row[21] if len(row) > 21 else '',
            'updated_at': row[22] if len(row) > 22 else ''
        }
    
    def _row_to_note_dict(self, row):
        """Convert database row to note dictionary"""
        return {
            'id': row[0],
            'candidate_id': row[1],
            'note_type': row[2],
            'note_text': row[3],
            'added_by': row[4],
            'created_at': row[5]
        }
    
    def _row_to_interview_dict(self, row, include_name=False):
        """Convert database row to interview dictionary"""
        result = {
            'id': row[0],
            'candidate_id': row[1],
            'round': row[2],
            'interviewer': row[3],
            'result': row[4],
            'rating': row[5],
            'feedback': row[6],
            'interview_date': row[7]
        }
        
        if include_name and len(row) > 8:
            result['candidate_name'] = row[8]
        
        return result


# Global instance
db = DatabaseManager()
