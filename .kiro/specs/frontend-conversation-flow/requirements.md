# Requirements Document

## Introduction

This feature implements the complete frontend conversation flow for the Lumi emotion detection application. The system enables children with neurological disabilities to interact with an AI-powered interface that detects their emotions through a multi-stage problem identification process. The workflow starts with emotion detection, generates initial problem prompts, allows drilling down to identify the exact problem through multiple levels, and provides context-aware solutions based on their profile and daily routine. Users can regenerate problem lists if their issue isn't listed and can request better solutions if not satisfied.

## Glossary

- **Lumi System**: The emotion detection and AI conversation application
- **Child Profile**: User profile containing child's name, age, and diagnosis information
- **Child Routine**: Daily schedule and activities for the child stored in the database
- **Session**: A single interaction instance from emotion detection to solution delivery
- **Conversation History**: Array of user selections throughout the interaction
- **Interaction Depth**: Counter tracking how many levels deep the conversation has gone
- **Initial Prompts**: First set of 4 AI-generated problem identification options shown after emotion detection
- **Follow-up Prompts**: Second level of 4 prompts shown after initial selection to narrow down the exact problem
- **Dig Deeper Prompts**: Additional sets of 4 prompts generated when user wants more exploration at any level
- **Problem Regeneration**: Generating a new set of 4 alternative problem options when the user's issue isn't listed
- **Solution**: Final AI-generated actionable response to address the child's emotional state
- **Gemini Service**: Backend API service that generates prompts and solutions using Google's Gemini AI
- **Supabase**: Database service storing child profiles, routines, and session data

## Requirements

### Requirement 1: Session State Management

**User Story:** As a parent, I want the system to maintain conversation context throughout the session, so that the AI can provide coherent and relevant responses.

#### Acceptance Criteria

1. WHEN the Lumi System initializes a session, THE Lumi System SHALL create state variables for child routine, conversation history, interaction depth, action button visibility, solution visibility, and solution content
2. WHEN a user makes a selection, THE Lumi System SHALL append the selection to the conversation history array
3. WHEN the interaction progresses, THE Lumi System SHALL increment the interaction depth counter
4. WHEN a session completes, THE Lumi System SHALL reset all state variables to initial values

### Requirement 2: Child Context Loading

**User Story:** As a parent, I want the system to load my child's profile and routine data, so that AI responses are personalized and contextually relevant.

#### Acceptance Criteria

1. WHEN the profile loads, THE Lumi System SHALL fetch the child routine from the child_routines table using the child profile ID
2. WHEN the child routine is retrieved, THE Lumi System SHALL store the routine data in component state
3. IF the routine fetch fails, THEN THE Lumi System SHALL log the error and continue with null routine data
4. WHEN making API calls to Gemini Service, THE Lumi System SHALL include child profile data containing child name, age, and diagnosis
5. WHEN making API calls to Gemini Service, THE Lumi System SHALL include the complete child routine data

### Requirement 3: Context-Aware Initial Problem Identification

**User Story:** As a child user, I want to see 4 relevant problem options based on my current emotion and daily schedule, so that I can identify what's bothering me.

#### Acceptance Criteria

1. WHEN fetching initial prompts, THE Lumi System SHALL send the detected emotion to the Gemini Service
2. WHEN fetching initial prompts, THE Lumi System SHALL send the current time in HH:MM 24-hour format to the Gemini Service
3. WHEN fetching initial prompts, THE Lumi System SHALL send the child profile object to the Gemini Service
4. WHEN fetching initial prompts, THE Lumi System SHALL send the child routine object to the Gemini Service
5. WHEN the Gemini Service returns prompts, THE Lumi System SHALL display exactly 4 problem identification picture cards
6. WHEN the Gemini Service generates initial prompts, THE Gemini Service SHALL create 4 distinct problem hypotheses based on emotion, routine, and current time context

### Requirement 4: Multi-Level Problem Refinement Flow

**User Story:** As a child user, I want to drill down through multiple levels to identify my exact problem, so that I can get the most accurate help.

#### Acceptance Criteria

1. WHEN a user selects an initial problem card, THE Lumi System SHALL fetch 4 follow-up prompts from the Gemini Service to narrow down the exact problem
2. WHEN a user selects a follow-up prompt card, THE Lumi System SHALL hide the picture cards and display action buttons
3. WHEN action buttons are displayed, THE Lumi System SHALL show both "Proceed to Solution" and "Dig Deeper" options
4. WHEN the user clicks "Dig Deeper", THE Lumi System SHALL call the dig-deeper API endpoint with full conversation context
5. WHEN dig deeper prompts are received, THE Lumi System SHALL display 4 new picture cards and increment interaction depth
6. WHEN the Gemini Service generates follow-up or dig deeper prompts, THE Gemini Service SHALL create exactly 4 cards attempting to identify the exact problem based on conversation history

### Requirement 5: Problem List Regeneration

**User Story:** As a child user, I want to generate a new set of problem options if my issue isn't listed, so that I can find the right problem category.

#### Acceptance Criteria

1. WHEN displaying initial problem cards, THE Lumi System SHALL provide an option to generate a new set of problems
2. WHEN the user requests a new problem list, THE Lumi System SHALL call the regenerate-problems API endpoint with the detected emotion and context
3. WHEN new problems are generated, THE Lumi System SHALL replace the current 4 problem cards with 4 new alternative problem hypotheses
4. WHEN the Gemini Service regenerates problems, THE Gemini Service SHALL create 4 different problem options that were not in the previous set
5. WHEN regenerating problems, THE Lumi System SHALL maintain the same emotion and context but explore different problem angles

### Requirement 6: Solution Generation and Delivery

**User Story:** As a parent, I want the system to provide actionable solutions based on the conversation, so that I can help address my child's needs effectively.

#### Acceptance Criteria

1. WHEN the user clicks "Proceed to Solution", THE Lumi System SHALL call the solution generation API endpoint with complete conversation history
2. WHEN the solution is received, THE Lumi System SHALL hide action buttons and display the solution component
3. WHEN displaying the solution, THE Lumi System SHALL show the solution text along with satisfaction feedback buttons
4. WHEN displaying the solution, THE Lumi System SHALL include the detected emotion context
5. IF solution generation fails, THEN THE Lumi System SHALL display an error message and allow retry

### Requirement 7: Solution Satisfaction and Session Reset

**User Story:** As a parent, I want to provide feedback on whether the solution helps and start a new session, so that the system can track effectiveness and be ready for the next interaction.

#### Acceptance Criteria

1. WHEN the user clicks "This Helps", THE Lumi System SHALL update the session record with the solution and satisfaction status as "satisfied"
2. WHEN the user clicks "This Helps", THE Lumi System SHALL create a session summary containing child name, emotion, conversation path, and solution
3. WHEN the user clicks "This Helps", THE Lumi System SHALL set session status to "completed" and record the end timestamp
4. WHEN the user clicks "This Helps", THE Lumi System SHALL reset the session to start state, clearing all conversation history and UI components
5. WHEN the user clicks "Not Satisfied", THE Lumi System SHALL call the solution regeneration API endpoint with the previous solution and conversation context
6. WHEN a regenerated solution is received, THE Lumi System SHALL update the displayed solution and allow the user to provide feedback again
7. WHEN the Gemini Service regenerates a solution, THE Gemini Service SHALL create a different solution approach that addresses the same problem from a new angle

### Requirement 8: Session Persistence

**User Story:** As a parent, I want all interactions to be saved in the database, so that I can review patterns and track my child's emotional journey over time.

#### Acceptance Criteria

1. WHEN a session is created, THE Lumi System SHALL store the session ID in component state
2. WHEN making dig deeper or solution requests, THE Lumi System SHALL include the session ID in API calls
3. WHEN a session completes successfully, THE Lumi System SHALL persist conversation history, interaction depth, solution, and satisfaction status to the database
4. WHEN a session completes, THE Lumi System SHALL record the ended_at timestamp
5. WHEN a new session starts after completion, THE Lumi System SHALL generate a new session ID

### Requirement 9: API Integration

**User Story:** As a developer, I want the frontend to correctly integrate with all Gemini Service endpoints, so that the AI features function reliably.

#### Acceptance Criteria

1. WHEN calling the initial prompts endpoint, THE Lumi System SHALL send a POST request to http://localhost:8001/api/prompts/initial with emotion, child profile, routine, and current time
2. WHEN calling the problem regeneration endpoint, THE Lumi System SHALL send a POST request to http://localhost:8001/api/prompts/regenerate-problems with session ID, emotion, previous problems, child profile, routine, and current time
3. WHEN calling the follow-up prompts endpoint, THE Lumi System SHALL send a POST request to http://localhost:8001/api/prompts/followup with session ID and selected option
4. WHEN calling the dig deeper endpoint, THE Lumi System SHALL send a POST request to http://localhost:8001/api/prompts/dig-deeper with session ID, emotion, conversation history, child profile, routine, and current time
5. WHEN calling the solution generation endpoint, THE Lumi System SHALL send a POST request to http://localhost:8001/api/solution/generate with session ID, emotion, conversation history, child profile, and routine
6. WHEN calling the solution regeneration endpoint, THE Lumi System SHALL send a POST request to http://localhost:8001/api/solution/regenerate with session ID, emotion, conversation history, previous solution, child profile, and routine

### Requirement 10: Loading States and User Feedback

**User Story:** As a child user, I want to see clear feedback when the system is processing, so that I know the application is working and not frozen.

#### Acceptance Criteria

1. WHEN fetching prompts from any endpoint, THE Lumi System SHALL set the loading state to true
2. WHILE the loading state is true, THE Lumi System SHALL display a loading indicator to the user
3. WHEN prompts or solutions are received, THE Lumi System SHALL set the loading state to false
4. WHEN an API call fails, THE Lumi System SHALL display an appropriate error message
5. WHEN displaying action buttons or solutions, THE Lumi System SHALL hide loading indicators

### Requirement 11: Component Integration

**User Story:** As a developer, I want to properly integrate the ActionButtons and SolutionDisplay components, so that the UI is cohesive and functional.

#### Acceptance Criteria

1. WHEN action buttons should be displayed, THE Lumi System SHALL render the ActionButtons component with onProceedToSolution and onDigDeeper callback props
2. WHEN the solution should be displayed, THE Lumi System SHALL render the SolutionDisplay component with solution text, emotion, onSatisfied, and onNotSatisfied callback props
3. WHEN action buttons are shown, THE Lumi System SHALL hide picture cards
4. WHEN the solution is shown, THE Lumi System SHALL hide action buttons
5. WHEN a new session starts, THE Lumi System SHALL hide both action buttons and solution display components
