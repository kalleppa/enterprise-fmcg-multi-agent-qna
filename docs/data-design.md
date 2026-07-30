# FMCG Data Design

## 1. Synthetic Company

The prototype uses a fictional FMCG company named **Nova Consumer Products**.

Nova Consumer Products manufactures and sells products across the following categories:

* Home Care
* Personal Care
* Packaged Foods

The company operates across multiple regions, states, sales channels, distributors, brands, products, and campaigns.

## 2. Business Hierarchies

### Product hierarchy

```text
Company
└── Category
    └── Brand
        └── Product
            └── SKU
```

Example:

```text
Nova Consumer Products
└── Home Care
    └── FreshGlow
        └── FreshGlow Lemon Dishwash
            └── FG-DW-LEM-500
```

### Geography hierarchy

```text
Country
└── Region
    └── State
        └── City
```

Example:

```text
India
└── South Region
    └── Karnataka
        └── Bengaluru
```

### Time hierarchy

```text
Year
└── Quarter
    └── Month
```

### Sales hierarchy

```text
Sales Channel
└── Distributor
```

## 3. Brands

The synthetic company contains three brands.

| Brand     | Category       | Description                               |
| --------- | -------------- | ----------------------------------------- |
| FreshGlow | Home Care      | Dishwashing and surface-cleaning products |
| PureHome  | Personal Care  | Personal hygiene and body-care products   |
| NutriBite | Packaged Foods | Healthy snacks and breakfast products     |

## 4. Products and SKUs

| SKU ID        | Product Name                       | Brand     | Category       | Pack Size |
| ------------- | ---------------------------------- | --------- | -------------- | --------- |
| FG-DW-LEM-500 | FreshGlow Lemon Dishwash           | FreshGlow | Home Care      | 500 ml    |
| FG-DW-LEM-1L  | FreshGlow Lemon Dishwash           | FreshGlow | Home Care      | 1 litre   |
| FG-SC-LAV-500 | FreshGlow Lavender Surface Cleaner | FreshGlow | Home Care      | 500 ml    |
| FG-SC-LAV-1L  | FreshGlow Lavender Surface Cleaner | FreshGlow | Home Care      | 1 litre   |
| PH-HW-ALO-250 | PureHome Aloe Handwash             | PureHome  | Personal Care  | 250 ml    |
| PH-HW-ALO-500 | PureHome Aloe Handwash             | PureHome  | Personal Care  | 500 ml    |
| PH-BW-NEE-500 | PureHome Neem Body Wash            | PureHome  | Personal Care  | 500 ml    |
| NB-GR-CHO-250 | NutriBite Chocolate Granola        | NutriBite | Packaged Foods | 250 g     |
| NB-GR-HON-500 | NutriBite Honey Granola            | NutriBite | Packaged Foods | 500 g     |
| NB-PB-CHO-6P  | NutriBite Chocolate Protein Bars   | NutriBite | Packaged Foods | Pack of 6 |

More products may be added during dataset generation.

## 5. Geographies

The prototype initially covers four regions.

| Region       | States                |
| ------------ | --------------------- |
| South Region | Karnataka, Tamil Nadu |
| West Region  | Maharashtra, Gujarat  |
| North Region | Delhi, Uttar Pradesh  |
| East Region  | West Bengal, Odisha   |

Example cities include:

* Bengaluru
* Chennai
* Mumbai
* Ahmedabad
* Delhi
* Lucknow
* Kolkata
* Bhubaneswar

## 6. Sales Channels

The following sales channels will be supported:

| Channel Code | Channel Name  |
| ------------ | ------------- |
| GT           | General Trade |
| MT           | Modern Trade  |
| EC           | E-commerce    |

The channel codes will also be included in the business glossary so that user queries such as `GT sales` can be mapped to `General Trade`.

## 7. Distributors

Example distributors include:

| Distributor ID | Distributor Name              | Region       |
| -------------- | ----------------------------- | ------------ |
| DST-SOUTH-01   | Sunrise Distributors          | South Region |
| DST-SOUTH-02   | Southern Retail Supply        | South Region |
| DST-WEST-01    | Western Trade Partners        | West Region  |
| DST-NORTH-01   | Capital Distribution Services | North Region |
| DST-EAST-01    | Eastern Consumer Supply       | East Region  |

## 8. Structured Datasets

The following structured datasets will be created.

### 8.1 Product dimension

File:

```text
data/structured/products.csv
```

Columns:

* sku_id
* product_name
* brand
* category
* subcategory
* pack_size
* launch_date
* discontinuation_date
* status

### 8.2 Geography dimension

File:

```text
data/structured/geography.csv
```

Columns:

* geography_id
* country
* region
* state
* city

### 8.3 Distributor dimension

File:

```text
data/structured/distributors.csv
```

Columns:

* distributor_id
* distributor_name
* region
* state
* channel
* active_status

### 8.4 Sales fact

File:

```text
data/structured/sales.csv
```

Columns:

* month
* sku_id
* geography_id
* distributor_id
* channel
* units_sold
* gross_revenue_inr
* discount_inr
* net_revenue_inr
* cogs_inr
* gross_margin_inr
* gross_margin_pct

### 8.5 Inventory fact

File:

```text
data/structured/inventory.csv
```

Columns:

* month
* sku_id
* geography_id
* distributor_id
* opening_stock_units
* received_units
* units_sold
* closing_stock_units
* stockout_days
* inventory_status

### 8.6 Promotion fact

File:

```text
data/structured/promotions.csv
```

Columns:

* campaign_id
* campaign_name
* sku_id
* geography_id
* channel
* start_date
* end_date
* promotion_spend_inr
* planned_sales_lift_pct
* actual_sales_lift_pct
* campaign_status

## 9. Unstructured Documents

The following documents will be created:

```text
data/documents/sparkle-summer-2025-campaign-brief.md
data/documents/q2-2025-category-review.md
data/documents/south-region-distributor-review.md
data/documents/inventory-exception-report.md
data/documents/freshglow-product-launch.md
data/documents/pricing-policy-2025.md
data/documents/product-discontinuation-notice.md
```

These documents will intentionally mention the same entities that appear in the structured datasets.

Example shared entities include:

* FreshGlow
* FG-DW-LEM-500
* South Region
* Karnataka
* Sunrise Distributors
* Sparkle Summer 2025
* Q2 2025

## 10. Controlled Business Scenarios

The generated data will include deliberate business scenarios so that the agent can answer analytical questions.

### Scenario 1: Successful promotion

A promotion for NutriBite exceeds its planned sales-lift target.

### Scenario 2: Underperforming campaign

The Sparkle Summer 2025 campaign has:

* Planned sales lift: 15%
* Actual sales lift: approximately 7%
* High promotion spend
* Inventory shortages in Karnataka

### Scenario 3: Recurring stockouts

SKU `FG-DW-LEM-500` experiences recurring stockouts in South Region.

### Scenario 4: Revenue growth with margin decline

West Region records revenue growth while gross-margin percentage declines because of higher discounts.

### Scenario 5: Product discontinuation

An older PureHome SKU is discontinued but remains mentioned in historical documents.

### Scenario 6: Distributor underperformance

Sunrise Distributors performs below its quarterly target because of stock availability and delayed replenishment.

## 11. Overlapping Evidence Example

The question:

> Did the Sparkle Summer 2025 campaign achieve its planned sales lift, and what affected the result?

may require information from:

* `promotions.csv` for planned and actual sales lift
* `sales.csv` for revenue and units sold
* `inventory.csv` for stockout information
* `sparkle-summer-2025-campaign-brief.md` for campaign objectives
* `q2-2025-category-review.md` for management explanations

This overlap allows the system to demonstrate hybrid retrieval and multi-agent synthesis.

## 12. Data Period

The initial dataset will cover:

```text
January 2024 to June 2025
```

This supports:

* Monthly analysis
* Quarterly analysis
* Year-over-year comparisons
* Historical comparisons
* Campaign-period analysis

## 13. Currency and Units

Financial values will use Indian rupees.

Examples:

* Gross revenue: INR
* Net revenue: INR
* Cost of goods sold: INR
* Promotion spend: INR

Product quantities will use:

* Units
* Millilitres
* Litres
* Grams
* Packs

The final agent response must always present appropriate units.

## 14. Data Quality Rules

The generated data must follow these rules:

* Every SKU must exist in the product dimension.
* Every geography ID must exist in the geography dimension.
* Every distributor ID must exist in the distributor dimension.
* Net revenue must equal gross revenue minus discount.
* Gross margin must equal net revenue minus cost of goods sold.
* Closing inventory must not be negative.
* Campaign dates must be valid.
* Discontinued products must have a discontinuation date.
* Percentages must remain within realistic ranges.
* Missing values must be intentional and documented.
