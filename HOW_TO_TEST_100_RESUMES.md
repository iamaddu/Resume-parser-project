# How to Test with 100 Resumes

## Quick Test Steps

### 1. Open the Application
- Go to: http://localhost:8501
- You should see "NEUROMATCH AI" header
- Sidebar shows: Home, Single Analysis, Bulk Processing

### 2. Navigate to Bulk Processing
- Click "Bulk Processing" in the sidebar
- You'll see the bulk processing page

### 3. Set Job Requirements
- **Required Skills:** python, sql, machine learning
- **Minimum Experience:** 3 years
- **Job Title:** Data Scientist
- **Minimum Education:** bachelor

### 4. Load 100 Test Resumes
- Select "Paste Text (Multiple Resumes)" radio button
- Open the file: `100_TEST_RESUMES.txt`
- Copy the ENTIRE file content (Ctrl+A, Ctrl+C)
- Paste into the text area
- Click "PROCESS TEXT INPUT" button

### 5. Verify Processing
- Progress bar should appear
- Status should show: "Processing profile 1/100", "2/100", etc.
- Should complete in ~30 seconds
- Final message: "Quantum processing complete!"

### 6. Check Results

#### KPI Metrics (Top)
- **Total Profiles:** Should show **100** (not a static number!)
- **Neural Selected:** Count of candidates with 80%+ match
- **Cyber Shortlisted:** Count of candidates with 65-79% match
- **Success Rate:** Percentage of selected candidates

#### Detailed Analysis
- Scroll down to see "DETAILED CANDIDATE ANALYSIS"
- Click on any candidate expander
- Should show:
  - Match Score, Experience, Skills, Status
  - Score Breakdown (6 progress bars)
  - Skills Analysis (matched/missing)
  - Risk Assessment
  - Compensation Analysis (4 salary metrics)
  - Selection Reasoning
  - Interview Questions (5 questions)
  - Email Template

#### Comparative Table
- Should show all 100 candidates
- Columns: Rank, Candidate, Match %, Experience, Skills, Skills Match, Status, Decision, Salary Range
- Data should be accurate (not fake/static)

#### Visual Analytics
- **Bar Chart:** Shows match scores for all candidates
  - X-axis: Candidate names
  - Y-axis: Match Score (0-100)
  - Colors: Status-based (cyan, magenta, yellow, red)
- **Scatter Plot:** Experience vs Skills
  - X-axis: Experience (years)
  - Y-axis: Skills count
  - Bubble size: Match score

#### Export Options
- **Export Summary CSV:** Downloads CSV with all 100 candidates
- **Export Detailed Report:** Downloads markdown report
- **Export All Emails:** Downloads 100 email templates

---

## What to Verify

### ✅ Dynamic Data (NOT Static)
- Total Profiles = **100** (actual count from your input)
- Selected count = actual number with 80%+ score
- Shortlisted count = actual number with 65-79% score
- Success rate = calculated from actual results

### ✅ Accurate Processing
- All 100 resumes processed
- Each has unique name (Sarah Chen, Michael Rodriguez, etc.)
- Scores vary based on actual resume content
- Experience ranges from 0-18 years
- Skills count varies per candidate

### ✅ Charts Work
- Bar chart shows all candidates (may need to scroll/zoom)
- Scatter plot shows distribution
- No errors or blank charts

### ✅ Exports Work
- CSV contains 100 rows
- Report lists all 100 candidates
- Email bundle has 100 separate emails

---

## Expected Results Distribution

With the 100 test resumes provided:

- **Senior/Expert (7+ years):** ~30 candidates → Should score 75-95%
- **Mid-level (3-6 years):** ~40 candidates → Should score 60-80%
- **Junior (1-2 years):** ~20 candidates → Should score 45-65%
- **Entry/Transition (<1 year):** ~10 candidates → Should score 30-50%

**Approximate Distribution:**
- Neural Selected (80%+): ~15-25 candidates
- Cyber Shortlisted (65-79%): ~30-40 candidates
- Processing (45-64%): ~25-35 candidates
- Filtered Out (<45%): ~5-15 candidates

---

## Troubleshooting

### Issue: Shows less than 100 candidates
**Cause:** Some resumes failed to parse or were empty
**Solution:** Check the console for errors, verify all resumes have content

### Issue: Processing hangs
**Cause:** Too many resumes or system resources
**Solution:** Try with 50 resumes first, then scale up

### Issue: Charts don't display
**Cause:** Browser compatibility or Plotly not installed
**Solution:** 
```bash
pip install plotly
```
Refresh browser (Ctrl+Shift+R)

### Issue: Static numbers still showing
**Cause:** Old code cached
**Solution:** Hard refresh (Ctrl+Shift+R) or restart app

---

## Performance Benchmarks

- **10 resumes:** ~3 seconds
- **50 resumes:** ~15 seconds
- **100 resumes:** ~30 seconds
- **200 resumes:** ~60 seconds

If processing takes significantly longer, check system resources.

---

## Success Criteria

✅ **You're good if:**
1. Total Profiles shows **100**
2. All KPI metrics are calculated (not static)
3. Detailed analysis available for all 100
4. Table shows all 100 rows
5. Charts render correctly
6. Exports contain 100 entries
7. Processing completes without errors
8. Results make logical sense

---

## Next Steps

After successful 100-resume test:

1. **Test with your own resumes** (real PDFs or text)
2. **Adjust job requirements** to see how scores change
3. **Export and review** CSV/reports
4. **Test interview questions** - are they personalized?
5. **Check email templates** - do they match candidate status?
6. **Verify salary predictions** - are they logical?

---

## Notes

- The 100 test resumes are **realistic and diverse**
- They include: seniors, mid-level, juniors, career changers, specialists
- Skills range from basic to expert
- Experience ranges from 0 to 18 years
- Education levels vary (bachelor, master, PhD)
- Some have red flags (job hopping, gaps)

This ensures comprehensive testing of all features!
