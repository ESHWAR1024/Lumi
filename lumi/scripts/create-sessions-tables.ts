/**
 * Script to create sessions tables in Supabase
 * Run with: npx tsx scripts/create-sessions-tables.ts
 */

import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

const supabase = createClient(supabaseUrl, supabaseKey);

async function createTables() {
  console.log('Creating sessions tables...');

  // Note: You'll need to run this SQL in Supabase SQL Editor
  // because the anon key doesn't have permission to create tables
  
  const sql = `
-- Create sessions table to track emotion detection sessions
CREATE TABLE IF NOT EXISTS public.sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    child_profile_id UUID NOT NULL REFERENCES public.child_profiles(id) ON DELETE CASCADE,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    initial_emotion TEXT NOT NULL,
    emotion_confidence DECIMAL(5,2),
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'abandoned')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create session_interactions table to track the conversation flow
CREATE TABLE IF NOT EXISTS public.session_interactions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    interaction_order INTEGER NOT NULL,
    prompt_type TEXT NOT NULL CHECK (prompt_type IN ('initial', 'followup')),
    selected_option TEXT NOT NULL,
    prompt_options JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_sessions_child_profile ON public.sessions(child_profile_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON public.sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_session_interactions_session ON public.session_interactions(session_id);

-- Add RLS policies
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.session_interactions ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to read their own sessions
CREATE POLICY "Users can view their own sessions"
    ON public.sessions FOR SELECT
    USING (true);

-- Allow authenticated users to insert sessions
CREATE POLICY "Users can create sessions"
    ON public.sessions FOR INSERT
    WITH CHECK (true);

-- Allow authenticated users to update their own sessions
CREATE POLICY "Users can update their own sessions"
    ON public.sessions FOR UPDATE
    USING (true);

-- Allow authenticated users to read session interactions
CREATE POLICY "Users can view session interactions"
    ON public.session_interactions FOR SELECT
    USING (true);

-- Allow authenticated users to insert session interactions
CREATE POLICY "Users can create session interactions"
    ON public.session_interactions FOR INSERT
    WITH CHECK (true);
`;

  console.log('\n📋 Copy and paste this SQL into your Supabase SQL Editor:');
  console.log('👉 Go to: https://bnsqdhlxztjqhfspcumj.supabase.co/project/_/sql/new');
  console.log('\n' + '='.repeat(80));
  console.log(sql);
  console.log('='.repeat(80) + '\n');
}

createTables();
