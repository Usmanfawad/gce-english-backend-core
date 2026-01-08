# Supabase Database Setup Guide

This directory contains SQL schema files and documentation for the CorpusAI Backend Core database.

## 📋 Prerequisites

1. **Supabase Account**: Create an account at [supabase.com](https://supabase.com)
2. **Project Created**: Set up a new Supabase project
3. **Database Access**: Have admin access to the SQL Editor

## 🚀 Setup Steps

### 1. Create Supabase Project

1. Go to [app.supabase.com](https://app.supabase.com)
2. Click "New Project"
3. Fill in:
   - **Project Name**: `corpusai-backend` (or your preferred name)
   - **Database Password**: Generate a strong password (save it securely!)
   - **Region**: Choose closest to your users
4. Wait for project provisioning (~2 minutes)

### 2. Get Connection String

#### Option A: Direct Connection (Recommended for Backend)
1. Go to **Settings** → **Database**
2. Under "Connection string", select **URI**
3. Copy the connection string
4. Format: `postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres`
5. For direct connection (no pooling), use port `5432`:
   ```
   postgresql://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres?sslmode=require
   ```

#### Option B: Connection Pooling (For Production)
- Use port `6543` (Transaction mode) or `5432` (Session mode)
- Pooling is better for high-concurrency applications

### 3. Enable Audit Logs

Supabase has built-in audit logging, but we also use our custom `audit_logs` table.

**Supabase Native Audit:**
1. Go to **Settings** → **Database** → **Logs**
2. Enable **Statement Logging** (optional, for query debugging)
3. Enable **PostgREST Logs** (API request logging)

**Custom Audit Logs (Our WORM Table):**
- Automatically enforced via triggers (see `worm_triggers.sql`)
- Run `worm_triggers.sql` in SQL Editor to enable WORM enforcement
- Prevents UPDATE and DELETE operations at database level

### 4. Create Tables and Enable WORM Triggers

1. Go to **SQL Editor** in Supabase Dashboard
2. Click **New Query**
3. Tables are automatically created by SQLModel on application startup
4. **Add Missing Columns** (If needed):
   - If `audit_logs` table exists but is missing `metadata_json` column, run `migration_add_metadata_json.sql`
   - Copy the contents of `migration_add_metadata_json.sql`
   - Paste into the SQL Editor and click **Run**
5. **Enable WORM Triggers** (Required for audit logs):
   - Copy the contents of `worm_triggers.sql`
   - Paste into the SQL Editor
   - Click **Run** to create triggers that prevent UPDATE/DELETE on audit_logs
6. Verify triggers were created:
   ```sql
   SELECT trigger_name, event_manipulation 
   FROM information_schema.triggers 
   WHERE event_object_table = 'audit_logs';
   ```
   You should see: `audit_logs_prevent_update` and `audit_logs_prevent_delete`
   ```sql
   SELECT table_name 
   FROM information_schema.tables 
   WHERE table_schema = 'public' 
   AND table_name IN ('users', 'user_connectors', 'audit_logs');
   ```

### 5. Configure Environment Variables

#### Local Development (`.env`)
```env
DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres?sslmode=require
```

#### GitHub Actions / CI/CD
1. Go to your repository → **Settings** → **Secrets and variables** → **Actions**
2. Add secret:
   - **Name**: `DATABASE_URL`
   - **Value**: Your connection string (redacted in logs)

#### Vercel Deployment
1. Go to your Vercel project → **Settings** → **Environment Variables**
2. Add variable:
   - **Key**: `DATABASE_URL`
   - **Value**: Your connection string
   - **Environments**: Production, Preview, Development (as needed)

### 6. Verify Database Health Check

Test the backend health endpoint:
```bash
curl http://localhost:8000/health/db
```

Expected response:
```json
{
  "status": "ok",
  "db": "available",
  "message": "Database connection healthy"
}
```

## 🔒 WORM (Write Once Read Many) Configuration

The `audit_logs` table is configured as **immutable** for compliance:

### What is WORM?
- **Write Once**: Records can be inserted, but never modified
- **Read Many**: Records can be queried indefinitely
- **Purpose**: Tamper-proof audit trail for compliance (SOC 2, GDPR, etc.)

### How It's Enforced

1. **Database Triggers**: Prevent UPDATE and DELETE operations
   ```sql
   -- Attempting UPDATE raises exception:
   UPDATE audit_logs SET action = 'modified' WHERE id = '...';
   -- ERROR: Audit logs are immutable. Updates are not allowed.
   ```

2. **Row Level Security** (Optional, recommended):
   - Enable RLS on `audit_logs` table
   - Users can only SELECT their own audit logs
   - INSERT must be done via service role

### Usage Example

#### Python Implementation

```python
from app.utils.audit import create_audit_log

# In your API endpoint
async def create_resource(session: AsyncSession, request: Request):
    # ... create resource ...
    
    # Log the action
    await create_audit_log(
        session=session,
        action="resource.created",
        user_id=current_user.id,  # if authenticated
        resource_type="resource",
        resource_id=resource.id,
        request=request,
        metadata={"additional": "data"}
    )
    
    await session.commit()
```

#### Direct SQL (for reference)

```sql
-- ✅ INSERT is allowed
INSERT INTO audit_logs (user_id, action, resource_type, resource_id)
VALUES ('user-uuid', 'user.created', 'user', 'user-uuid');

-- ❌ UPDATE will fail
UPDATE audit_logs SET action = 'modified' WHERE id = '...';
-- → ERROR: Audit logs are immutable. Updates are not allowed.

-- ❌ DELETE will fail
DELETE FROM audit_logs WHERE id = '...';
-- → ERROR: Audit logs are immutable. Deletes are not allowed.
```

### Automatic Audit Logging

The application automatically logs all API requests via middleware:
- All HTTP requests (except health endpoints) are logged
- Includes IP address, user agent, method, path, status code
- Errors are logged with error messages
- Performance metrics (process time) included in metadata

### Best Practices

1. **Always Log**: Record all critical actions (user creation, API calls, errors)
2. **Never Skip**: Don't bypass audit logging for "performance"
3. **Structured Data**: Use `metadata` JSONB field for additional context
4. **Retention**: Plan for long-term storage (audit logs never expire)

## 📊 Table Schemas

### `users`
- User accounts and authentication
- Fields: `id`, `email`, `username`, `hashed_password`, `is_active`, etc.
- Indexed on `email` for fast lookups

### `user_connectors`
- External API connections per user (OpenAI, Anthropic, etc.)
- Stores encrypted API keys
- Supports JSONB config for flexible connector settings
- One user can have multiple connectors of different types

### `audit_logs`
- **Immutable** audit trail
- Records all user actions, API calls, and system events
- Includes IP address, user agent, request metadata
- Indexed for efficient querying by user, action, date

## 🔐 Security Notes

1. **Connection Strings**: Never commit to Git
   - Use environment variables
   - Rotate passwords regularly
   - Use connection pooling in production

2. **API Keys**: Store encrypted in `user_connectors`
   - Never log or expose API keys
   - Use Supabase encryption or your own encryption layer

3. **Row Level Security**: Enable RLS for multi-tenant scenarios
   - Users should only access their own data
   - Service role for administrative operations

4. **Audit Logs**: Service role access only for INSERT
   - Regular users can only SELECT their own logs
   - Prevents tampering with audit trail

## 🧪 Testing

### Test Database Connection
```bash
# From your backend
curl http://localhost:8000/health/db
```

### Test Audit Logging
```sql
-- Insert a test audit log
INSERT INTO audit_logs (action, resource_type, metadata)
VALUES ('test.action', 'test', '{"test": true}'::jsonb);

-- Verify it exists
SELECT * FROM audit_logs WHERE action = 'test.action';

-- Try to update (should fail)
UPDATE audit_logs SET action = 'modified' WHERE action = 'test.action';
-- ERROR: Audit logs are immutable. Updates are not allowed.

-- Try to delete (should fail)
DELETE FROM audit_logs WHERE action = 'test.action';
-- ERROR: Audit logs are immutable. Deletes are not allowed.
```

## 📝 Migration Strategy

When updating schema:
1. Create migration files: `supabase/migrations/YYYYMMDD_description.sql`
2. Test in staging environment first
3. Apply to production during maintenance window
4. Verify data integrity after migration

## 🆘 Troubleshooting

### Connection Issues
- Verify connection string format
- Check SSL mode (`sslmode=require` for Supabase)
- Verify password is correct
- Check IP allowlist in Supabase Dashboard

### Table Creation Errors
- Ensure UUID extension is enabled
- Check for syntax errors in SQL
- Verify you have CREATE TABLE permissions

### Audit Log Errors
- Verify triggers were created successfully
- Check that WORM policies are active
- Ensure service role has INSERT permissions

## 📚 References

- [Supabase Documentation](https://supabase.com/docs)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [AsyncPG Documentation](https://magicstack.github.io/asyncpg/)

## ✅ Acceptance Criteria Checklist

- [x] Supabase project created
- [x] Connection string documented
- [x] Tables created: `users`, `user_connectors`, `audit_logs`
- [x] WORM configuration enforced (triggers)
- [x] Database health check endpoint (`/health/db`) working
- [x] Connection strings configured in GitHub/Vercel (documented)
- [x] Audit logs append-only functionality verified




