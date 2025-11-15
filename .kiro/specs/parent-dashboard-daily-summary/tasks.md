# Implementation Plan

- [x] 1. Create database schema for daily summaries


  - Create daily_summaries table in Supabase with proper columns and constraints
  - Add unique constraint on (child_profile_id, date)
  - Create index for faster queries
  - _Requirements: 1.1, 4.1, 4.4_

- [ ] 2. Implement backend API endpoint for daily summary generation
- [ ] 2.1 Create /api/summary/daily endpoint in gemini_service/app.py


  - Add POST endpoint that accepts child_profile_id, date, session_summaries, and child_name
  - Implement request validation
  - Add error handling for missing parameters
  - _Requirements: 6.1, 6.2, 6.4_

- [ ] 2.2 Create generate_daily_summary function in gemini_prompts.py
  - Write Gemini prompt for daily summary generation
  - Implement function to call Gemini API with session summaries
  - Use separate GEMINI_SUMMARY_API_KEY environment variable (fallback to main key if not set)
  - Format and return the generated summary
  - Handle API errors gracefully
  - _Requirements: 4.1, 4.2, 4.3, 6.3_

- [ ] 2.3 Integrate summary generation with database storage
  - Store generated summary in daily_summaries table
  - Return success response with summary text
  - Handle duplicate summary attempts (date already exists)
  - _Requirements: 4.4_

- [ ] 3. Update frontend start page with Parent's Dashboard
- [ ] 3.1 Add "Parent's Dashboard" heading to hamburger menu
  - Add heading above child name in menu
  - Style heading to be visually distinct
  - Position at top of menu content area
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 3.2 Replace "Weekly Reports" with "Daily Summary" section
  - Remove existing Weekly Reports section
  - Add new Daily Summary section with 📊 icon
  - Maintain consistent styling with other menu sections
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 3.3 Create DailySummary interface and state management
  - Define DailySummary TypeScript interface
  - Add state for daily summaries array
  - Add loading state for summaries
  - _Requirements: 3.1_

- [ ] 4. Implement daily summary fetching and display
- [ ] 4.1 Create function to fetch daily summaries from database
  - Query daily_summaries table for child's recent summaries
  - Limit to most recent 7 days
  - Order by date descending
  - Handle empty results gracefully
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 4.2 Create function to generate missing daily summaries
  - Check if yesterday's summary exists when page loads
  - Fetch all completed sessions for the previous day
  - Call backend API to generate summary if sessions exist
  - Store result in database
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 4.3 Create DailySummaryCard component
  - Display date in readable format (e.g., "Monday, Jan 15")
  - Show summary text with proper formatting
  - Display session count badge
  - Add expand/collapse for long summaries
  - _Requirements: 3.1, 3.2_

- [ ] 4.4 Integrate summary display into hamburger menu
  - Map daily summaries to DailySummaryCard components
  - Show loading skeleton while fetching
  - Display "No summaries yet" message when empty
  - Add error handling for fetch failures
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 5. Add useEffect hook to trigger summary generation on page load
  - Call function to check and generate missing summaries
  - Only run once on component mount
  - Handle errors silently (don't block user experience)
  - _Requirements: 5.1, 5.2_

- [ ] 6. Add helper functions for date formatting and session queries
  - Create function to format dates for display
  - Create function to get yesterday's date
  - Create function to query sessions by date and child
  - _Requirements: 3.2, 4.2, 5.2_

- [ ] 7. Configure separate API key for daily summaries
  - Add GEMINI_SUMMARY_API_KEY to .env file in gemini_service
  - Update .env.example with the new variable
  - Document the purpose of separate API keys in comments
  - Implement fallback to main GEMINI_API_KEY if summary key not set
  - _Requirements: 6.1, 6.4_
