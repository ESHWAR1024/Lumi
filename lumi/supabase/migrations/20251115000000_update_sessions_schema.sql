-- Add session_summary column to sessions table
ALTER TABLE public.sessions 
ADD COLUMN IF NOT EXISTS session_summary TEXT,
ADD COLUMN IF NOT EXISTS solution_provided TEXT,
ADD COLUMN IF NOT EXISTS satisfaction_status TEXT CHECK (satisfaction_status IN ('satisfied', 'not_satisfied', 'pending'));

-- Update session_interactions to support deeper conversation flow
ALTER TABLE public.session_interactions
ADD COLUMN IF NOT EXISTS interaction_depth INTEGER DEFAULT 1,
ADD COLUMN IF NOT EXISTS parent_interaction_id UUID REFERENCES public.session_interactions(id),
ADD COLUMN IF NOT EXISTS action_type TEXT CHECK (action_type IN ('initial', 'followup', 'dig_deeper', 'solution', 'confirmation'));

-- Add index for parent_interaction_id
CREATE INDEX IF NOT EXISTS idx_session_interactions_parent ON public.session_interactions(parent_interaction_id);

-- Add comments for clarity
COMMENT ON COLUMN public.sessions.session_summary IS 'Final summary of the session after child confirms satisfaction';
COMMENT ON COLUMN public.sessions.solution_provided IS 'AI-generated solution to the child''s problem';
COMMENT ON COLUMN public.sessions.satisfaction_status IS 'Whether child was satisfied with the solution';
COMMENT ON COLUMN public.session_interactions.interaction_depth IS 'How deep in the conversation tree (1=initial, 2=first followup, etc.)';
COMMENT ON COLUMN public.session_interactions.parent_interaction_id IS 'Reference to parent interaction for conversation tree';
COMMENT ON COLUMN public.session_interactions.action_type IS 'Type of interaction in the flow';
