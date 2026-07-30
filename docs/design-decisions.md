# Design Decisions

## 1. Project Objective

This project implements an enterprise-level question-answering agent prototype for a synthetic FMCG company.

The system uses a main orchestration agent and the following specialist sub-agents:

* Structured Data Retrieval Agent
* Unstructured Document Retrieval Agent
* Internet Search Agent
* Coding and Analysis Agent

The main agent identifies the user’s intent, selects the appropriate sub-agent or combination of sub-agents, gathers evidence, validates the results, and generates the final response.

## 2. Synthetic FMCG Company

The prototype uses a fictional FMCG company named **Nova Consumer Products**.

The company operates across multiple:

* Brands
* Products
* SKUs
* Categories
* Regions
* States
* Sales channels
* Distributors
* Campaigns
* Time periods

The same entities will appear in structured datasets and unstructured documents.

This overlap allows the agent to answer questions that require information from one or multiple sources.

Example entities include:

| Entity Type | Examples                       |
| ----------- | ------------------------------ |
| Brand       | FreshGlow, PureHome, NutriBite |
| Product     | FreshGlow Lemon Dishwash       |
| SKU         | FG-DW-LEM-500                  |
| Category    | Home Care                      |
| Region      | South Region                   |
| State       | Karnataka                      |
| Channel     | General Trade                  |
| Distributor | Sunrise Distributors           |
| Campaign    | Sparkle Summer 2025            |

## 3. Multi-Agent Architecture

A multi-agent architecture was selected because each type of task requires different tools, validation rules, and retrieval strategies.

The main agent acts as the orchestrator.

It is responsible for:

1. Understanding the user query.
2. Validating whether the request is supported.
3. Detecting missing or ambiguous information.
4. Asking a clarification question when required.
5. Selecting one or more specialist agents.
6. Collecting results from the specialist agents.
7. Combining structured and unstructured evidence.
8. Validating the final answer.
9. Formatting the response with citations and assumptions.
10. Maintaining conversation context.

## 4. Structured Data Retrieval Agent

The structured data agent answers questions using tabular FMCG data.

Example questions include:

* What was the revenue for FreshGlow in Q2 2025?
* Which region had the highest sales?
* Compare gross margin across brands.
* Which SKU had the most stockout days?

The agent will:

1. Read the available database metadata.
2. Identify relevant tables and columns.
3. Generate a SQL query.
4. Validate the SQL query.
5. Execute the query using a read-only connection.
6. Return the result with units, periods, and source information.

DuckDB will initially be used because it is lightweight, local, reproducible, and suitable for a prototype.

A production implementation could replace DuckDB with PostgreSQL, Redshift, Snowflake, Databricks, or another enterprise data platform.

## 5. SQL Safety

Generated SQL will be restricted to read-only queries.

The following SQL operations will not be allowed:

* INSERT
* UPDATE
* DELETE
* DROP
* ALTER
* TRUNCATE
* CREATE
* GRANT
* REVOKE

Additional safety controls include:

* Allowlisted tables
* Single-statement execution
* Query timeout
* Maximum returned row count
* SQL parsing before execution
* Read-only database connection
* Logging of generated SQL
* Rejection of suspicious SQL patterns

The SQL safety layer reduces risk but does not replace database-level access control.

## 6. Unstructured Document Retrieval Agent

The document retrieval agent answers questions using FMCG documents such as:

* Campaign briefs
* Product launch documents
* Distributor reviews
* Pricing policies
* Inventory reports
* Quarterly business reviews
* Product discontinuation notices

Documents will be divided into section-aware chunks.

Each chunk will contain metadata such as:

* Document title
* Document type
* Brand
* Product
* SKU
* Region
* State
* Campaign
* Effective date
* Tags
* Source path
* Section name

The agent will use hybrid retrieval by combining:

* Semantic vector search
* Keyword search
* Metadata filtering
* Result ranking
* Duplicate removal

Every document-based answer will include source citations.

## 7. Internet Search Agent

The internet search agent will be used only when current or external information is required.

Example requests include:

* Find recent FMCG market trends.
* Search for current commodity-price information.
* Compare internal results with current public market information.

Internet search will not be used automatically for every question.

This decision reduces unnecessary latency, cost, and irrelevant external information.

Internal and external information will be clearly separated in the final response.

## 8. Coding and Analysis Agent

The coding agent will perform calculations and data analysis that are difficult to handle through retrieval alone.

Example tasks include:

* Percentage-change calculations
* Correlation analysis
* Trend analysis
* Data-quality checks
* Chart generation
* Forecast demonstrations
* Statistical summaries

The coding agent will receive approved data returned by the structured retrieval agent.

It will not receive unrestricted database credentials.

The coding environment should have:

* Execution timeout
* Restricted package access
* No unrestricted network access
* Controlled file access
* Error capture
* Result validation

## 9. Intent Validation and Routing

The system will classify requests into the following intent categories:

* Greeting
* Capability query
* Metadata query
* Structured-data query
* Document query
* Hybrid query
* Internet-search query
* Coding or analytical query
* Clarification required
* Unsupported request

Simple rules will be used before calling a language model.

This reduces unnecessary model calls and improves predictability.

A language model will be used when rule-based routing is not confident enough.

## 10. Clarification Handling

The agent will ask a clarification question when missing information could materially change the answer.

Examples include:

* Missing time period
* Ambiguous product
* Missing region
* Missing KPI
* Missing comparison period
* Unsupported granularity

Example:

> Which period should I use: Q1 2025, Q2 2025, or the latest available quarter?

The agent should ask one focused clarification question at a time.

## 11. Conversation Memory

The agent will support multi-turn conversations.

Conversation state will include:

* Recent messages
* Resolved entities
* Selected KPI
* Selected time period
* Selected geography
* Previous intent
* Previous result summary
* Preferred language
* Pending clarification

Recent conversation turns will remain available directly.

Older conversation turns will be summarized to reduce token usage and latency.

Large SQL results and complete documents will not be repeatedly added to the model context.

## 12. Semantic Understanding

The agent will use a business glossary for aliases, abbreviations, spelling variations, and multilingual terms.

Examples include:

| User Input       | Canonical Value |
| ---------------- | --------------- |
| Fresh Glow       | FreshGlow       |
| GT               | General Trade   |
| KA               | Karnataka       |
| rev              | Revenue         |
| बिक्री           | Sales           |
| South region में | South Region    |

Exact aliases will be resolved first.

Fuzzy matching will be used only when the confidence is sufficiently high.

Low-confidence matches will require clarification.

## 13. Hybrid Retrieval

Some questions require evidence from multiple sources.

Example:

> Did the Sparkle Summer campaign achieve its planned sales lift, and why?

This question may require:

* Campaign target from the promotion dataset
* Actual sales from the sales dataset
* Inventory information from the inventory dataset
* Explanation from a campaign or quarterly-review document
* Coding agent calculation of actual sales lift

For hybrid questions, independent retrieval tasks may run in parallel to reduce latency.

## 14. Answer Validation

Before returning the final answer, the system will validate:

* Whether numerical values match tool outputs
* Whether units are present
* Whether periods are clearly stated
* Whether important claims have supporting evidence
* Whether document citations are included
* Whether assumptions are reported
* Whether internal and external evidence are separated
* Whether tool errors are transparently reported

If validation fails, the system may retry retrieval or regeneration.

Retries will be limited to avoid uncontrolled cost and latency.

## 15. Model Usage Strategy

The same model should not be used for every task.

Suggested model strategy:

| Task                       | Preferred Approach                    |
| -------------------------- | ------------------------------------- |
| Greeting                   | Static response                       |
| Capability introduction    | Static response                       |
| Intent classification      | Rules or small model                  |
| Entity normalization       | Glossary, fuzzy matching, small model |
| SQL generation             | Small or medium reasoning model       |
| Document query expansion   | Small model                           |
| Simple answer synthesis    | Small or medium model                 |
| Complex hybrid synthesis   | Stronger model                        |
| Conversation summarization | Small model                           |
| Validation                 | Deterministic checks first            |

Stronger and more expensive models will be reserved for complex multi-source synthesis.

## 16. Cost Strategy

The system will control cost through:

* Deterministic routing
* Model routing
* Metadata caching
* Embedding caching
* SQL-result caching
* Small document-retrieval limits
* Conversation summarization
* Bounded retries
* Static responses for greetings
* Selective internet search
* Calling only required agents

For each request, the prototype should capture:

* Input tokens
* Output tokens
* Number of model calls
* Number of tool calls
* Estimated model cost
* Total response latency
* Per-agent latency

Model prices will be configurable and will not be hard-coded into the implementation.

## 17. Latency Strategy

Latency will be reduced by:

* Running hybrid retrieval tasks in parallel
* Caching database metadata
* Caching document embeddings
* Limiting retrieved document chunks
* Using deterministic logic when possible
* Applying query timeouts
* Limiting retry attempts
* Avoiding unnecessary internet searches
* Avoiding unnecessary model calls

Suggested prototype targets are:

| Request Type            |   Target Latency |
| ----------------------- | ---------------: |
| Greeting                |   Under 1 second |
| Metadata query          |  Under 2 seconds |
| Structured query        |  Under 5 seconds |
| Document query          |  Under 6 seconds |
| Hybrid query            | Under 10 seconds |
| Coding query            | Under 12 seconds |
| Internet-assisted query | Under 15 seconds |

These are prototype targets and not production guarantees.

## 18. Document Trade-Offs

Larger document chunks preserve more context but increase model-token usage and may reduce retrieval precision.

Smaller chunks improve retrieval precision but may separate related statements.

The prototype will use section-aware chunks with limited overlap.

Metadata will be stored with every chunk so results can be filtered by brand, product, region, date, document type, and tags.

The number of retrieved chunks will remain limited to control cost and latency.

## 19. Synthetic Data Trade-Offs

Synthetic data provides the following advantages:

* Safe to publish
* Reproducible
* Easy to test
* Supports controlled business scenarios
* Enables known expected answers
* Allows deliberate entity overlap

Its limitations include:

* It does not represent full enterprise scale.
* It may contain fewer inconsistencies than production data.
* It cannot prove production-level performance.
* Entity resolution may be easier than with real data.

These limitations will be clearly documented.

## 20. Key Design Principle

The main design principle is:

> Use deterministic logic where possible and language-model reasoning where necessary.

This approach provides better control over cost, latency, security, predictability, and answer quality.
