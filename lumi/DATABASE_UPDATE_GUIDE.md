# Database Update Guide - Add "Can Type" Field

## What Changed

Added a new field `can_type` to the `child_profiles` table to track whether a child can type on a laptop/keyboard.

---

## How to Apply the Migration

### Option 1: Using Supabase Dashboard (Easiest)

1. Go to your Supabase project dashboard
2. Click on **SQL Editor** in the left sidebar
3. Click **New Query**
4. Copy and paste the following SQL:

```sql
-- Add can_type column to child_profiles table
ALTER TABLE public.child_profiles 
ADD COLUMN IF NOT EXISTS can_type BOOLEAN DEFAULT true;

-- Add comment for clarity
COMMENT ON COLUMN public.child_profiles.can_type IS 'Indicates whether the child can type on a laptop/keyboard (true = yes, false = no)';

-- Update existing records to have default value of true
UPDATE public.child_profiles 
SET can_type = true 
WHERE can_type IS NULL;
```

5. Click **Run** or press `Ctrl+Enter`
6. You should see "Success. No rows returned"

---

### Option 2: Using Supabase CLI

```bash
# Make sure you're in the lumi directory
cd lumi

# Run the migration
supabase db push
```

---

## Verify the Migration

### Check in Supabase Dashboard:

1. Go to **Table Editor**
2. Select `child_profiles` table
3. You should see a new column called `can_type` (type: boolean)

### Check Existing Data:

All existing profiles will have `can_type = true` by default.

---

## What This Enables

- Parents can now specify if their child can type during onboarding
- This information can be used to:
  - Customize the UI (show/hide keyboard input options)
  - Adjust interaction methods
  - Provide alternative input methods for non-typing children
  - Generate more appropriate solutions based on typing ability

---

## Frontend Changes

The onboarding form now includes:
- Radio buttons for "Yes" / "No" selection
- Default value: "Yes" (true)
- Stored in the database when profile is created

---

## For Your Friends

When your friends pull the latest code, they need to:

1. **Pull the latest code** from GitHub
2. **Run the database migration** (Option 1 or 2 above)
3. **Restart the frontend** if it's already running

That's it! The new field will be available for all new profiles.

---

## Rollback (If Needed)

If you need to remove this field:

```sql
ALTER TABLE public.child_profiles 
DROP COLUMN IF EXISTS can_type;
```

**Note:** This will permanently delete the data in this column!
