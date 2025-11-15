# Implementation Plan

- [x] 1. Fix TypeScript type definitions and null safety issues





  - Update ChildProfile interface to include diagnosis field
  - Add proper null checks for childProfile throughout the component
  - Fix all TypeScript errors in start/page.tsx
  - _Requirements: All requirements depend on proper type safety_
-

- [x] 2. Implement child routine loading in profile initialization




  - [x] 2.1 Update loadProfile() to fetch child routine data


    - Add Supabase query to fetch from child_routines table
    - Store routine data in childRoutine state
    - Handle case where routine doesn't exist (redirect to /routine)
    - _Requirements: 2.1, 2.2, 2.3_
  

  - [x] 2.2 Add getCurrentTime() utility function

    - Create function that returns current time in HH:MM 24-hour format
    - Use toLocaleTimeString with proper options
    - _Requirements: 3.2_

- [x] 3. Update initial prompts API integration with full context





  - [x] 3.1 Modify fetchInitialPrompts() to include child context


    - Add child_profile object with child_name, age, diagnosis
    - Add child_routine object
    - Add current_time from getCurrentTime()
    - Keep existing emotion and confidence fields
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 8.1_
  
  - [x] 3.2 Update createSession() to store initial session data


    - Ensure session record includes all required fields
    - Handle session creation errors gracefully
    - _Requirements: 7.1, 7.2_

- [x] 4. Implement conversation history tracking






  - [x] 4.1 Add conversationHistory state management

    - Initialize as empty array
    - Update handleCardSelect() to append selections
    - Maintain history throughout session lifecycle
    - _Requirements: 1.2, 4.1, 4.2_
  
  - [x] 4.2 Add interactionDepth state tracking

    - Initialize to 1
    - Increment when dig deeper is used
    - Reset on new session
    - _Requirements: 1.3, 4.5_

- [x] 5. Update card selection logic for multi-level flow





  - [x] 5.1 Modify handleCardSelect() to route based on promptType


    - If promptType is "initial", fetch follow-up prompts
    - If promptType is "followup", show action buttons
    - Add selection to conversation history before routing
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [x] 5.2 Update fetchFollowupPrompts() with full context


    - Include interaction_depth in request body
    - Include conversation_history in request body
    - Include child_profile, child_routine, current_time
    - _Requirements: 8.2_
  
  - [x] 5.3 Update storeInteraction() to track depth


    - Add interaction_depth field to database insert
    - Ensure proper interaction ordering
    - _Requirements: 7.3_

- [x] 6. Implement dig deeper functionality




  - [x] 6.1 Create handleDigDeeper() function

    - Hide action buttons and show loading state
    - Call /api/prompts/dig-deeper endpoint with full context
    - Include session_id, emotion, conversation_history
    - Include child_profile, child_routine, current_time
    - Update prompts state with response
    - Show cards again and increment interaction depth
    - Store interaction in database
    - _Requirements: 4.4, 4.5, 8.3_
  
  - [x] 6.2 Wire handleDigDeeper to ActionButtons component

    - Pass function as onDigDeeper prop
    - Ensure proper error handling
    - _Requirements: 10.1_

- [-] 7. Implement solution generation


  - [x] 7.1 Create handleProceedToSolution() function


    - Hide action buttons and show loading state
    - Call /api/solution/generate endpoint
    - Include session_id, emotion, conversation_history
    - Include child_profile and child_routine
    - Update solution state with response
    - Show solution display component
    - Update session record with solution
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 8.4_
  
  - [x] 7.2 Wire handleProceedToSolution to ActionButtons component





    - Pass function as onProceedToSolution prop
    - Ensure proper error handling
    - _Requirements: 10.1_

- [x] 8. Implement solution satisfaction feedback






  - [x] 8.1 Create handleSatisfied() function

    - Create session summary with child name, emotion, conversation path, and solution
    - Update session with solution, satisfaction_status as "satisfied"
    - Set session status to "completed" and record ended_at timestamp
    - Show success message to user
    - Reset all state variables for new session
    - _Requirements: 6.1, 6.2, 6.3, 7.4_
  

  - [x] 8.2 Create handleNotSatisfied() function

    - Show loading state
    - Call /api/solution/regenerate endpoint
    - Include session_id, emotion, conversation_history, previous_solution
    - Include child_profile and child_routine
    - Update solution state with new response
    - Update session record with new solution
    - _Requirements: 6.4, 6.5, 8.5_
  

  - [x] 8.3 Wire feedback handlers to SolutionDisplay component

    - Pass handleSatisfied as onSatisfied prop
    - Pass handleNotSatisfied as onNotSatisfied prop
    - _Requirements: 10.2_

- [x] 9. Implement UI state management and transitions




  - [x] 9.1 Add showActionButtons state and logic


    - Initialize to false
    - Set to true after follow-up card selection
    - Set to false when dig deeper or proceed to solution clicked
    - _Requirements: 4.3, 10.3_
  
  - [x] 9.2 Add showSolution state and logic


    - Initialize to false
    - Set to true when solution is generated
    - Set to false on new session
    - _Requirements: 5.2, 10.4_
  
  - [x] 9.3 Update conditional rendering for all UI phases


    - Ensure proper AnimatePresence transitions
    - Hide cards when action buttons show
    - Hide action buttons when solution shows
    - Show loading state during API calls
    - _Requirements: 9.1, 9.2, 9.3, 10.3, 10.4_

- [x] 10. Implement comprehensive error handling





  - [x] 10.1 Add error handling for all API calls



    - Wrap fetch calls in try-catch blocks
    - Set appropriate error messages for different failure types
    - Clear loading states in finally blocks
    - _Requirements: 5.5, 9.4_
  
  - [x] 10.2 Add error handling for database operations


    - Handle profile not found scenarios
    - Handle routine not found scenarios
    - Log errors to console for debugging
    - _Requirements: 2.3_
  

  - [x] 10.3 Improve user feedback for errors

    - Display clear error messages
    - Provide retry options where appropriate
    - Ensure error state is cleared on new operations
    - _Requirements: 9.4, 9.5_

- [x] 11. Add loading states for better UX






  - [x] 11.1 Ensure loadingPrompts state is properly managed

    - Set to true before all API calls
    - Set to false after responses received
    - Set to false in error cases
    - _Requirements: 9.1, 9.2, 9.3_
  

  - [x] 11.2 Update conditional rendering for loading states

    - Show loading spinner when loadingPrompts is true
    - Hide other UI elements during loading
    - Display "Thinking..." message
    - _Requirements: 9.5_

- [x] 12. Implement session reset functionality






  - [x] 12.1 Create resetSession() helper function

    - Reset emotion and confidence
    - Clear conversation history
    - Reset interaction depth to 1
    - Clear prompts and solution
    - Hide all UI components (cards, actions, solution)
    - Clear session ID
    - _Requirements: 1.4, 7.5, 10.5_
  
  - [x] 12.2 Call resetSession() after successful completion

    - Integrate into handleSatisfied()
    - Ensure clean state for next session
    - _Requirements: 1.4_

- [x] 13. Verify complete integration and flow





  - Test camera to emotion detection flow
  - Test initial prompts with context awareness
  - Test follow-up prompts generation
  - Test dig deeper functionality
  - Test solution generation
  - Test satisfaction feedback loop
  - Test regenerate solution
  - Test session persistence
  - Verify all TypeScript errors are resolved
  - _Requirements: All requirements_
