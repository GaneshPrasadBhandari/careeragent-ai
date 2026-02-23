CareerAgent-AI: 10-Layer Strategic Blueprint
Layer 0: User Layer (The Interface)
Purpose: Captures user goals, constraints (US location, 36h recency), and identity documents (Resume).


Agents: * Intake Assistant: Collects target roles and salary expectations. 


Evaluator (L0): Validates that all critical fields are filled before triggering the pipeline. 
+1


Current implementation: Streamlit-based UI. 

Layer 1: Entry Point Layer (Security & Sanitization)
Purpose: Ensures the system is secure and inputs are clean.


Agents: * Sanitize Agent: Screens inputs for prompt injection and anonymizes sensitive PII. 
+2


Evaluator (L1): Confirms the session is secure and API keys are active. 
+1

Layer 2: Orchestrator Core Layer (The Brain)
Purpose: Converts user intent into an executable multi-agent plan.

Agents: * Planner: Breaks the hunt into sub-tasks (Search, Scrape, Match).

Director: Reviews the plan to ensure it meets constraints (e.g., "Must be US-based").


Parser Agent: Extracts skills and experience from the resume. 
+1


Evaluator (L2): Scores the quality of the parsed profile. 
+1

Layer 3: Manager Layer (Discovery & Competitive Intelligence)
Purpose: Finds and filters high-quality job opportunities.

Agents (5 Managers):

Lead Scout: Generates search personas (e.g., "AI Architect") with negative keywords (e.g., "-India").


Search Ops: Executes queries via Serper/Tavily APIs. 


Geo-Fencer: Hard-filters results to ensure "United States" location. 

Extraction Manager: Uses Jina Reader to pull full job descriptions.

Deduplicator: Removes duplicate listings across platforms.


Evaluator (L3): Ensures the "Job Pool" is sufficient. 
+1

Layer 4: Agents Layer (Identity RAG & Matching)
Purpose: Deep analysis of how your identity fits each discovered job.


Agents: * Skill Matcher: Maps your 16+ years of experience to the JD. 
+1

Gap Analyst: Identifies missing keywords for ATS optimization.


RAG Agent: Retrieves project context from Qdrant/ChromaDB. 


Evaluator (L4): Scores matches; jobs below 0.60 are discarded. 
+1

Layer 5: Human-In-The-Loop (HITL Approval)
Purpose: Final human verification to prevent "Black Box" errors.


Operation: The system pauses at 60% progress. 


Agents: Liaison Agent presents the ranking for your approval. 
+1


Evaluator (L5): Validates your selections before moving to Execution. 
+1

Layer 6: Execution Layer (Auto-Apply & Outreach)
Purpose: Taking real-world action on approved jobs.


Agents: * Executive Drafter: Creates ATS-friendly tailored resumes and cover letters. 

Email Executive: Drafts cold outreach for hiring managers.

Auto-Applier: Uses Selenium/Playwright to fill forms (Workday/Greenhouse).

Evaluator (L6): Proofreads all drafts for hallucinations or formatting errors.

Layer 7: Analytics & Learning Layer (The Dashboard)
Purpose: Tracking success and updating the system's "Intelligence."


Agents: * Dashboard Manager: Tracks "Applied" status, dates, and priority. 

Self-Learning Agent: Ingests user/employer feedback to refine future searches.


Career Coach: Generates an upskilling roadmap based on "Gaps" found in L4. 

Evaluator (L7): Verifies the accuracy of the analytics.

Layer 8: Infrastructure Layer (DB & LLM Models)
Purpose: The foundation of data and connectivity.


Tools: * Vector DBs: Qdrant (Cloud) and Chroma (Local) for long-term memory. 

MCP Server: Facilitates tool-calling between agents and local files/calendars.

Evaluator (L8): Monitors model latency and token usage.

Layer 9: Governance, Ethics & Ops (The Auditor)
Purpose: Ensuring compliance and system health.

Agents: Compliance Auditor ensures the bot doesn't spam or violate TOS.

Observability: LangSmith for full-trace agentic transparency.

Evaluator (L9): Final "all-clear" before completing the run.

Gap Analysis: What to Fix Immediately
L6 Evaluator: We need to implement a specific check for ATS-friendly formatting (e.g., standard margins, keyword density) after the Tailor Agent finishes its work.

L2 Director: Your current system lacks a "Director" to reject the search plan if it defaults to global locations instead of the US-only constraint.

Persistence: The Dashboard (L7) must be backed by a stable SQLite database to store follow-up dates and interview statuses.