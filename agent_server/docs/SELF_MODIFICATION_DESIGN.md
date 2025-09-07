# Agent Self-Modification Design

## Overview

This document outlines the design for enabling the agent to modify its own source code, test changes, and deploy them safely.

## Core Principles

- **Sandbox Security**: Agent can only write to designated directories
- **Version Control**: All changes tracked via git
- **Test Validation**: Deploy only after tests pass
- **Human Oversight**: Test changes require human approval
- **Safe Deployment**: Backup and rollback capabilities

## Environment Configuration

### Environment Variables

```bash
# Core paths (development-friendly defaults)
AGENT_DATA_ROOT=./agent-data                        # Root for all agent data
AGENT_CODE_REPO_PATH=${AGENT_DATA_ROOT}/code-repo   # Git repo (agent writable)
AGENT_SANDBOX_PATH=${AGENT_DATA_ROOT}/sandbox       # Free playground (agent writable)

# Git configuration for agent commits
AGENT_GIT_USER_NAME="Agent"
AGENT_GIT_USER_EMAIL="agent@localhost"

# Deploy settings
AGENT_DEPLOY_REQUIRE_TESTS=true
AGENT_DEPLOY_TEST_TIMEOUT=300
AGENT_REQUIRE_HUMAN_APPROVAL_FOR_TEST_CHANGES=true
```

### Directory Structure

#### Development Setup
```
/home/user/projects/agent/
├── agent_server/          # Original source (read-only for agent)
├── client/                # Original source (read-only for agent)
├── agent-data/            # AGENT_DATA_ROOT
│   ├── code-repo/         # AGENT_CODE_REPO_PATH (git clone, agent writable)
│   │   ├── .git/
│   │   ├── agent_server/
│   │   ├── client/
│   │   └── [all source files]
│   ├── sandbox/           # AGENT_SANDBOX_PATH (empty start, agent writable)
│   ├── models/            # Server data (agent read-only)
│   ├── logs/              # Server data (agent read-only)
│   ├── generated_images/  # Server data (agent read-only)
│   ├── conversations/     # Server data (agent read-only)
│   └── deployments/       # Deploy metadata (server managed)
│       ├── backups/
│       ├── status.json
│       └── human_approval_required.flag
└── [other project files]
```

#### Production Setup
```
/opt/agent-data/           # AGENT_DATA_ROOT (override in production)
├── code-repo/             # Same structure as development
├── sandbox/
├── models/
├── logs/
├── generated_images/
├── conversations/
└── deployments/
```

## Permissions Model

### Agent Read Access
- **Everything**: Agent can read any file in the system
- Used for: Understanding codebase, analyzing logs, reading documentation

### Agent Write Access
- **code-repo/**: Git repository of agent's source code
- **sandbox/**: Free experimentation area
- **Nowhere else**: All other locations are read-only for the agent

### Server Write Access
- **models/**, **logs/**, **generated_images/**, **conversations/**: Server data
- **deployments/**: Deployment metadata and backups

## Git Workflow

### 1. Code Modification
- Agent modifies files in `code-repo/`
- All changes tracked in git working directory
- Agent can create branches, commits, etc.

### 2. Commit Process
```bash
cd ${AGENT_CODE_REPO_PATH}
git add [modified files]
git commit -m "Agent modification: [description]"
```

### 3. Deploy Action Workflow
1. **Test Validation**
   - Run full test suite in `code-repo/`
   - Timeout: `AGENT_DEPLOY_TEST_TIMEOUT` seconds
   - If tests fail → abort deployment

2. **Human Approval Check**
   - If any test files were modified → set `human_approval_required.flag`
   - Block deployment until human reviews and clears flag
   - Rationale: Agent shouldn't modify its own validation logic without oversight

3. **Backup Current Version**
   - Create backup of currently running code in `deployments/backups/`
   - Include timestamp and commit hash for rollback

4. **Hot Deploy**
   - Copy validated code from `code-repo/` to running server
   - For Python: Hot-reload modules (where safe)
   - For React: Rebuild and serve new client

5. **Git Push** (Optional)
   - If deployment successful and commit hooks pass
   - Push changes to remote repository
   - Preserves agent's evolution history

## Deploy Action Implementation

### New Action Type
```python
class DeployAgentCodeInput(BaseModel):
    commit_message: str = Field(description="Commit message for the changes")
    force_deploy: bool = Field(default=False, description="Skip human approval check")

class DeployAgentCodeOutput(ActionOutput):
    success: bool
    commit_hash: str
    tests_passed: bool
    human_approval_required: bool
    deployed: bool
    backup_location: str
```

### Safety Mechanisms
1. **Test Requirements**: All tests must pass before deployment
2. **Human Approval**: Required for any test file modifications
3. **Backup System**: Automatic backup before each deployment
4. **Rollback Capability**: Easy revert to previous version
5. **Commit Hooks**: Additional validation before git push

## Implementation Steps

1. **Environment Setup**
   - Add environment variable parsing to `paths.py`
   - Create initial directory structure
   - Set up git repository clone

2. **Permission System**
   - Modify file operations to respect write restrictions
   - Add validation for agent file access

3. **Deploy Action**
   - Implement `DeployAgentCodeAction` class
   - Add test runner integration
   - Create backup/restore functionality

4. **Human Approval System**
   - Flag mechanism for test modifications
   - Admin endpoint to approve/reject changes
   - UI for deployment oversight

5. **Hot Reload System**
   - Python module reloading for server code
   - Client rebuild and serving for frontend code
   - Graceful handling of reload failures

## Security Considerations

- **Write Restrictions**: Agent cannot modify core server data
- **Test Integrity**: Human oversight required for test changes
- **Rollback Safety**: Always maintain working backup
- **Git History**: Full audit trail of agent modifications
- **Commit Validation**: Optional commit hooks for additional safety

## Future Enhancements

- **Staged Deployment**: Test in isolated environment before production
- **Automatic Rollback**: Revert if deployed code fails health checks
- **Change Approval Workflow**: More sophisticated human review process
- **Remote Deployment**: Deploy to multiple agent instances
- **Diff Review UI**: Visual interface for reviewing agent changes