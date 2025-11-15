import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export interface ChildProfile {
  id?: string;
  child_name: string;
  age: number;
  parent_email: string;
  condition: string;
  diagnosis?: string;
  created_at?: string;
  updated_at?: string;
}

export interface ChildRoutine {
  id?: string;
  child_profile_id: string;
  wake_time: string;
  breakfast_time: string;
  lunch_time: string;
  dinner_time: string;
  snack_times: string[];
  nap_time: string;
  bedtime: string;
  favorite_activities: string[];
  comfort_items: string[];
  triggers_to_avoid: string[];
  preferred_prompts: string[];
  created_at?: string;
  updated_at?: string;
}
