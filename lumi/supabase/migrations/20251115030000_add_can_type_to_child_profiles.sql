-- Add can_type column to child_profiles table
-- This field indicates whether the child can type on a laptop/keyboard

ALTER TABLE public.child_profiles 
ADD COLUMN IF NOT EXISTS can_type BOOLEAN DEFAULT true;

-- Add comment for clarity
COMMENT ON COLUMN public.child_profiles.can_type IS 'Indicates whether the child can type on a laptop/keyboard (true = yes, false = no)';

-- Update existing records to have default value of true
UPDATE public.child_profiles 
SET can_type = true 
WHERE can_type IS NULL;
