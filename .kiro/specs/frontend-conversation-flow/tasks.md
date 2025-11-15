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





  - [x] 3.1 Modify fetchInitialPrompts() to include child context and ensure 4 cards


    - Add child_profile object with child_name, age, diagnosis
    - Add child_routine object
    - Add current_time from getCurrentTime()
    - Keep existing emotion and confidence fields
    - Verify response contains exactly 4 problem cards
    - Store problem labels in previousProblems state
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 9.1_
  
  - [x] 3.2 Update createSession() to store initial session data


    - Ensure session record includes all required fields
    - Handle session creation errors gracefully
    - _Requirements: 8.1, 8.2_

- [x] 4. Implement problem regeneration functionality



  - [x] 4.1 Add previousProblems state management


    - Initialize as empty array
    - Store problem labels after initial prompts fetch
    - Append new problems after regeneration
    - _Requirements: 5.2, 5.3, 5.4_

  - [x] 4.2 Create RegenerateButton component


    - Create new component file with props interface (onRegenerate, loading)
    - Implement orange/amber gradient styling
    - Add Framer Motion animations
    - Display text: "Not listed? Show me different problems"
    - _Requirements: 5.1_

  - [x] 4.3 Implement handleRegenerateProblems() function


    - Show loading state
    - Call /api/prompts/regenerate-problems endpoint
    - Include session_id, emotion, previous_problems, child_profile, child_routine, current_time
    - Update prompts state with 4 new cards
    - Append new problem labels to previousProblems
    - _Requirements: 5.2, 5.3, 5.4, 5.5, 9.2_



  - [x] 4.4 Add showRegenerateButton state and logic

    - Initialize to false
    - Set to true after initial prompts load
    - Set to false after card selection or when moving to follow-ups
    - Only show when promptType is "initial"
    - _Requirements: 5.1_

- [x] 5. Implement conversation history tracking






  - [x] 5.1 Add conversationHistory state management

    - Initialize as empty array
    - Update handleCardSelect() to append selections
    - Maintain history throughout session lifecycle
    - _Requirements: 1.2, 4.1, 4.2_
  
  - [x] 5.2 Add interactionDepth state tracking

    - Initialize to 1
    - Increment when dig deeper is used
    - Reset on new session
    - _Requirements: 1.3, 4.5, 4.6_

- [x] 6. Update card selection logic for multi-level flow





  - [x] 6.1 Modify handleCardSelect() to route based on promptType


    - If promptType is "initial", fetch follow-up prompts
    - If promptType is "followup", show action buttons
    - Add selection to conversation history before routing
    - Hide regenerate button after selection
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [x] 6.2 Update fetchFollowupPrompts() with full context and ensure 4 cards


    - Include interaction_depth in request body
    - Include conversation_history in request body
    - Include child_profile, child_routine, current_time
    - Verify response contains exactly 4 follow-up cards
    - _Requirements: 4.6, 9.3_
  
  - [x] 6.3 Update storeInteraction() to track depth


    - Add interaction_depth field to database insert
    - Ensure proper interaction ordering
    - _Requirements: 8.3_

- [x] 7. Implement dig deeper functionality




  - [x] 7.1 Create handleDigDeeper() function and ensure 4 cards

    - Hide action buttons and show loading state
    - Call /api/prompts/dig-deeper endpoint with full context
    - Include session_id, emotion, conversation_history
    - Include child_profile, child_routine, current_time
    - Verify response contains exactly 4 dig deeper cards
    - Update prompts state with response
    - Show cards again and increment interaction depth
    - Store interaction in database
    - _Requirements: 4.4, 4.5, 4.6, 9.4_
  
  - [x] 7.2 Wire handleDigDeeper to ActionButtons component

    - Pass function as onDigDeeper prop
    - Ensure proper error handling
    - _Requirements: 11.1_

- [-] 8. Implement solution generation


  - [x] 8.1 Create handleProceedToSolution() function


    - Hide action buttons and show loading state
    - Call /api/solution/generate endpoint
    - Include session_id, emotion, conversation_history
    - Include child_profile and child_routine
    - Update solution state with response
    - Show solution display component
    - Update session record with solution
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 9.5_
  
  - [x] 8.2 Wire handleProceedToSolution to ActionButtons component





    - Pass function as onProceedToSolution prop
    - Ensure proper error handling
    - _Requirements: 11.1_

- [x] 9. Implement solution satisfaction feedback and session reset






  - [x] 9.1 Create handleSatisfied() function with complete session reset

    - Create session summary with child name, emotion, conversation path, and solution
    - Update session with solution, satisfaction_status as "satisfied"
    - Set session status to "completed" and record ended_at timestamp
    - Reset ALL state variables: emotion, confidence, prompts, conversationHistory, interactionDepth, previousProblems
    - Hide all UI components (cards, action buttons, solution)
    - Return to start screen ready for new session
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.4_
  

  - [x] 9.2 Create handleNotSatisfied() function

    - Show loading state
    - Call /api/solution/regenerate endpoint
    - Include session_id, emotion, conversation_history, previous_solution
    - Include child_profile and child_routine
    - Update solution state with new response
    - Allow user to provide feedback again
    - Update session record with new solution
    - _Requirements: 7.5, 7.6, 7.7, 9.6_
  

  - [x] 9.3 Wire feedback handlers to SolutionDisplay component

    - Pass handleSatisfied as onSatisfied prop
    - Pass handleNotSatisfied as onNotSatisfied prop
    - Update button text to "This Helps" and "Not Satisfied"
    - _Requirements: 11.2_

- [x] 10. Implement UI state management and transitions




  - [x] 10.1 Add showActionButtons state and logic


    - Initialize to false
    - Set to true after follow-up card selection
    - Set to false when dig deeper or proceed to solution clicked
    - _Requirements: 4.3, 11.3_
  
  - [x] 10.2 Add showSolution state and logic


    - Initialize to false
    - Set to true when solution is generated
    - Set to false on new session
    - _Requirements: 6.2, 11.4_
  
  - [x] 10.3 Update conditional rendering for all UI phases


    - Ensure proper AnimatePresence transitions
    - Show RegenerateButton only when promptType is "initial" and showRegenerateButton is true
    - Hide cards when action buttons show
    - Hide action buttons when solution shows
    - Show loading state during API calls
    - _Requirements: 10.1, 10.2, 10.3, 11.3, 11.4_

- [x] 11. Implement comprehensive error handling





  - [x] 11.1 Add error handling for all API calls



    - Wrap fetch calls in try-catch blocks
    - Set appropriate error messages for different failure types
    - Clear loading states in finally blocks
    - Handle regenerate-problems endpoint errors
    - _Requirements: 6.5, 10.4_
  
  - [x] 11.2 Add error handling for database operations


    - Handle profile not found scenarios
    - Handle routine not found scenarios
    - Log errors to console for debugging
    - _Requirements: 2.3_
  

  - [x] 11.3 Improve user feedback for errors

    - Display clear error messages
    - Provide retry options where appropriate
    - Ensure error state is cleared on new operations
    - _Requirements: 10.4, 10.5_

- [x] 12. Add loading states for better UX






  - [x] 12.1 Ensure loadingPrompts state is properly managed

    - Set to true before all API calls including regenerate-problems
    - Set to false after responses received
    - Set to false in error cases
    - _Requirements: 10.1, 10.2, 10.3_
  

  - [x] 12.2 Update conditional rendering for loading states

    - Show loading spinner when loadingPrompts is true
    - Hide other UI elements during loading
    - Display "Thinking..." message
    - _Requirements: 10.5_

- [x] 13. Implement session reset functionality






  - [x] 13.1 Create resetSession() helper function

    - Reset emotion and confidence
    - Clear conversation history
    - Reset interaction depth to 1
    - Clear prompts and solution
    - Clear previousProblems array
    - Hide all UI components (cards, actions, solution, regenerate button)
    - Clear session ID
    - Return to start screen
    - _Requirements: 1.4, 7.4, 8.5, 11.5_
  
  - [x] 13.2 Call resetSession() after successful completion

    - Integrate into handleSatisfied()
    - Ensure clean state for next session
    - Verify user returns to start screen
    - _Requirements: 1.4, 7.4_

- [x] 14. Verify complete integration and flow







  - Test camera to emotion detection flow
  - Test initial prompts show exactly 4 cards
  - Test regenerate problems generates 4 new different cards
  - Test follow-up prompts show exactly 4 cards
  - Test dig deeper generates exactly 4 cards at each level
  - Test solution generation
  - Test "This Helps" resets session to start
  - Test "Not Satisfied" regenerates solution
  - Test session persistence
  - Verify all TypeScript errors are resolved
  - _Requirements: All requirements_
