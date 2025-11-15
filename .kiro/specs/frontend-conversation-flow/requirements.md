# Requirements Document

## Introduction

This feature implements the complete frontend conversation flow for the Lumi emotion detection application. The system enables children with neurological disabilities to interact with an AI-powered interface that detects their emotions, engages them in a multi-level conversation to understand their needs, and provides context-aware solutions based on their profile and daily routine.

## Glossary

- **Lumi System**: The emotion detection and AI conversation application
- **Child Profile**: User profile containing child's name, age, and diagnosis information
- **Child Routine**: Daily schedule and activities for the child stored in the database
- **Session**: A single interaction instance from emotion detection to solution delivery
- **Conversation History**: Array of user selections throughout the interaction
- **Interaction Depth**: Counter tracking how many levels deep the conversation has gone
- **Initial Prompts**: First set of AI-generated options shown after emotion detection
- **Follow-up Prompts**: Second level prompts shown after initial selection
- **Dig Deeper Prompts**: Additional prompts generated when user wants more exploration
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

### Requirement 3: Context-Aware Initial Prompts

**User Story:** As a child user, I want to see relevant options based on my current emotion and daily schedule, so that I can quickly express what I need.

#### Acceptance Criteria

1. WHEN fetching initial prompts, THE Lumi System SHALL send the detected emotion to the Gemini Service
2. WHEN fetching initial prompts, THE Lumi System SHALL send the current time in HH:MM 24-hour format to the Gemini Service
3. WHEN fetching initial prompts, THE Lumi System SHALL send the child profile object to the Gemini Service
4. WHEN fetching initial prompts, THE Lumi System SHALL send the child routine object to the Gemini Service
5. WHEN the Gemini Service returns prompts, THE Lumi System SHALL display them as selectable picture cards

### Requirement 4: Multi-Level Conversation Flow

**User Story:** As a child user, I want to navigate through different levels of conversation, so that I can explore my feelings and needs at my own pace.

#### Acceptance Criteria

1. WHEN a user selects an initial prompt card, THE Lumi System SHALL fetch follow-up prompts from the Gemini Service
2. WHEN a user selects a follow-up prompt card, THE Lumi System SHALL hide the picture cards and display action buttons
3. WHEN action buttons are displayed, THE Lumi System SHALL show both "Proceed to Solution" and "Dig Deeper" options
4. WHEN the user clicks "Dig Deeper", THE Lumi System SHALL call the dig-deeper API endpoint with full conversation context
5. WHEN dig deeper prompts are received, THE Lumi System SHALL display new picture cards and increment interaction depth

### Requirement 5: Solution Generation and Delivery

**User Story:** As a parent, I want the system to provide actionable solutions based on the conversation, so that I can help address my child's needs effectively.

#### Acceptance Criteria

1. WHEN the user clicks "Proceed to Solution", THE Lumi System SHALL call the solution generation API endpoint with complete conversation history
2. WHEN the solution is received, THE Lumi System SHALL hide action buttons and display the solution component
3. WHEN displaying the solution, THE Lumi System SHALL show the solution text along with satisfaction feedback buttons
4. WHEN displaying the solution, THE Lumi System SHALL include the detected emotion context
5. IF solution generation fails, THEN THE Lumi System SHALL display an error message and allow retry

### Requirement 6: Solution Satisfaction Feedback

**User Story:** As a parent, I want to provide feedback on whether the solution helps, so that the system can learn and improve or provide alternatives.

#### Acceptance Criteria

1. WHEN the user clicks "This Helps!", THE Lumi System SHALL update the session record with the solution and satisfaction status as "satisfied"
2. WHEN the user clicks "This Helps!", THE Lumi System SHALL create a session summary containing child name, emotion, conversation path, and solution
3. WHEN the user clicks "This Helps!", THE Lumi System SHALL set session status to "completed" and record the end timestamp
4. WHEN the user clicks "Try Again", THE Lumi System SHALL call the solution regeneration API endpoint with the previous solution
5. WHEN a regenerated solution is received, THE Lumi System SHALL update the displayed solution without changing other UI elements

### Requirement 7: Session Persistence

**User Story:** As a parent, I want all interactions to be saved in the database, so that I can review patterns and track my child's emotional journey over time.

#### Acceptance Criteria

1. WHEN a session is created, THE Lumi System SHALL store the session ID in component state
2. WHEN making dig deeper or solution requests, THE Lumi System SHALL include the session ID in API calls
3. WHEN a session completes successfully, THE Lumi System SHALL persist conversation history, interaction depth, solution, and satisfaction status to the database
4. WHEN a session completes, THE Lumi System SHALL record the ended_at timestamp
5. WHEN a new session starts after completion, THE Lumi System SHALL generate a new session ID

### Requirement 8: API Integration

**User Story:** As a developer, I want the frontend to correctly integrate with all Gemini Service endpoints, so that the AI features function reliably.

#### Acceptance Criteria

1. WHEN calling the initial prompts endpoint, THE Lumi System SHALL send a POST request to http://localhost:8001/api/prompts/initial with emotion, child profile, routine, and current time
2. WHEN calling the follow-up prompts endpoint, THE Lumi System SHALL send a POST request to http://localhost:8001/api/prompts/followup with session ID and selected option
3. WHEN calling the dig deeper endpoint, THE Lumi System SHALL send a POST request to http://localhost:8001/api/prompts/dig-deeper with session ID, emotion, conversation history, child profile, routine, and current time
4. WHEN calling the solution generation endpoint, THE Lumi System SHALL send a POST request to http://localhost:8001/api/solution/generate with session ID, emotion, conversation history, child profile, and routine
5. WHEN calling the solution regeneration endpoint, THE Lumi System SHALL send a POST request to http://localhost:8001/api/solution/regenerate with session ID, emotion, conversation history, previous solution, child profile, and routine

### Requirement 9: Loading States and User Feedback

**User Story:** As a child user, I want to see clear feedback when the system is processing, so that I know the application is working and not frozen.

#### Acceptance Criteria

1. WHEN fetching prompts from any endpoint, THE Lumi System SHALL set the loading state to true
2. WHILE the loading state is true, THE Lumi System SHALL display a loading indicator to the user
3. WHEN prompts or solutions are received, THE Lumi System SHALL set the loading state to false
4. WHEN an API call fails, THE Lumi System SHALL display an appropriate error message
5. WHEN displaying action buttons or solutions, THE Lumi System SHALL hide loading indicators

### Requirement 10: Component Integration

**User Story:** As a developer, I want to properly integrate the ActionButtons and SolutionDisplay components, so that the UI is cohesive and functional.

#### Acceptance Criteria

1. WHEN action buttons should be displayed, THE Lumi System SHALL render the ActionButtons component with onProceedToSolution and onDigDeeper callback props
2. WHEN the solution should be displayed, THE Lumi System SHALL render the SolutionDisplay component with solution text, emotion, onSatisfied, and onNotSatisfied callback props
3. WHEN action buttons are shown, THE Lumi System SHALL hide picture cards
4. WHEN the solution is shown, THE Lumi System SHALL hide action buttons
5. WHEN a new session starts, THE Lumi System SHALL hide both action buttons and solution display components
