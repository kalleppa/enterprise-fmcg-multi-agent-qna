from __future__ import annotations

from pathlib import Path
from textwrap import dedent


DOCUMENTS: dict[str, str] = {
    "sparkle-summer-2025-campaign-brief.md": dedent(
        """
        ---
        document_id: DOC-CAMPAIGN-001
        document_type: campaign_brief
        title: Sparkle Summer 2025 Campaign Brief
        brand: FreshGlow
        product: FreshGlow Lemon Dishwash
        sku_id: FG-DW-LEM-500
        campaign_id: CMP-2025-001
        campaign_name: Sparkle Summer 2025
        regions:
          - South Region
        states:
          - Karnataka
          - Tamil Nadu
        channels:
          - General Trade
          - Modern Trade
        effective_date: 2025-04-01
        tags:
          - campaign
          - sales-lift
          - promotion
          - freshglow
        ---

        # Sparkle Summer 2025 Campaign Brief

        ## Campaign objective

        The Sparkle Summer 2025 campaign is designed to increase sales of
        FreshGlow Lemon Dishwash 500 ml, SKU `FG-DW-LEM-500`, in the South
        Region.

        The campaign will run from 1 April 2025 to 30 June 2025.

        The primary objective is to achieve a **15% increase in unit sales**
        compared with the pre-campaign baseline.

        ## Target markets

        The campaign will focus on:

        - Karnataka
        - Tamil Nadu
        - General Trade
        - Modern Trade

        Sunrise Distributors will manage General Trade execution in Karnataka.
        Southern Retail Supply will support Modern Trade execution in Tamil
        Nadu.

        ## Investment

        The planned promotion investment is:

        | Market | Channel | Promotion spend |
        |---|---|---:|
        | Karnataka | General Trade | ₹25,00,000 |
        | Tamil Nadu | Modern Trade | ₹18,00,000 |

        ## Campaign activities

        Planned activities include:

        - Retailer display incentives
        - Consumer price discounts
        - In-store demonstrations
        - Distributor sales incentives
        - Regional digital advertising

        ## Key risks

        The campaign team identified the following risks:

        1. Insufficient inventory availability in Karnataka
        2. Delayed replenishment to Sunrise Distributors
        3. Margin pressure caused by promotional discounts
        4. Inconsistent retailer display execution
        5. Lower campaign awareness outside major cities

        ## Success measures

        The campaign will be considered successful when:

        - Unit sales lift reaches or exceeds 15%
        - Product availability remains above 95%
        - Stockout days remain below three days per month
        - Gross-margin percentage remains within the approved range
        """
    ).strip(),

    "q2-2025-category-review.md": dedent(
        """
        ---
        document_id: DOC-REVIEW-001
        document_type: quarterly_business_review
        title: Q2 2025 Home Care Category Review
        brand: FreshGlow
        category: Home Care
        campaign_id: CMP-2025-001
        campaign_name: Sparkle Summer 2025
        regions:
          - South Region
          - West Region
        effective_date: 2025-07-10
        tags:
          - quarterly-review
          - home-care
          - sales
          - inventory
          - margin
        ---

        # Q2 2025 Home Care Category Review

        ## Executive summary

        The Home Care category recorded revenue growth during Q2 2025, but
        performance varied significantly by region.

        FreshGlow remained the largest contributor to Home Care revenue.

        ## Sparkle Summer 2025 performance

        The Sparkle Summer 2025 campaign did not achieve its planned sales-lift
        target.

        The campaign targeted a 15% increase in unit sales for FreshGlow Lemon
        Dishwash 500 ml.

        Reported campaign lift was:

        - Karnataka General Trade: 6.8%
        - Tamil Nadu Modern Trade: 7.4%

        The result was below plan despite strong promotional visibility.

        ## Reasons for underperformance

        The review identified the following causes:

        1. FreshGlow Lemon Dishwash 500 ml experienced repeated stockouts in
           Karnataka.
        2. Sunrise Distributors received replenishment later than planned.
        3. General Trade execution was weaker in smaller cities.
        4. Promotional discounts improved revenue volume but reduced margin
           percentage.
        5. Some consumers shifted to the one-litre pack instead of the promoted
           500 ml pack.

        ## Inventory impact

        Karnataka experienced the most serious availability issues.

        Stockout days for SKU `FG-DW-LEM-500` were reported as:

        | Month | Stockout days |
        |---|---:|
        | April 2025 | 8 |
        | May 2025 | 11 |
        | June 2025 | 9 |

        Management concluded that inventory availability materially limited
        the campaign's sales potential.

        ## West Region observation

        West Region generated revenue growth during 2025.

        However, gross-margin percentage declined because of higher Modern
        Trade and E-commerce discounts.

        ## Recommended actions

        - Increase safety stock for promoted SKUs.
        - Confirm distributor inventory before campaign launch.
        - Reduce overlapping discount schemes.
        - Monitor pack-size substitution during promotions.
        - Create weekly inventory alerts for campaign products.
        """
    ).strip(),

    "south-region-distributor-review.md": dedent(
        """
        ---
        document_id: DOC-DISTRIBUTOR-001
        document_type: distributor_review
        title: South Region Distributor Performance Review
        region: South Region
        states:
          - Karnataka
          - Tamil Nadu
        distributors:
          - Sunrise Distributors
          - Southern Retail Supply
        effective_date: 2025-07-05
        tags:
          - distributor
          - south-region
          - performance
          - inventory
        ---

        # South Region Distributor Performance Review

        ## Review period

        This document reviews distributor performance from January 2025 to
        June 2025.

        ## Sunrise Distributors

        Sunrise Distributors manages General Trade distribution in Karnataka.

        The distributor performed below its Q2 target.

        The main issues were:

        - Delayed warehouse replenishment
        - Low availability of `FG-DW-LEM-500`
        - Uneven coverage outside Bengaluru
        - Slow retailer-order fulfilment
        - High dependence on manual inventory reporting

        Sunrise Distributors was particularly affected during the Sparkle
        Summer 2025 campaign.

        Although campaign demand increased, available inventory was not
        sufficient to meet the planned sales level.

        ## Southern Retail Supply

        Southern Retail Supply supports Modern Trade distribution in Tamil
        Nadu.

        Its service levels remained more stable than Sunrise Distributors, but
        campaign sales still remained below the 15% target.

        ## Corrective actions

        The following actions were agreed:

        1. Introduce weekly inventory forecasts.
        2. Increase safety stock for campaign SKUs.
        3. Track distributor fulfilment time.
        4. Implement automatic stockout alerts.
        5. Review General Trade coverage in smaller Karnataka cities.
        """
    ).strip(),

    "inventory-exception-report.md": dedent(
        """
        ---
        document_id: DOC-INVENTORY-001
        document_type: inventory_exception_report
        title: FreshGlow Inventory Exception Report
        brand: FreshGlow
        sku_id: FG-DW-LEM-500
        region: South Region
        state: Karnataka
        distributor: Sunrise Distributors
        effective_date: 2025-06-30
        tags:
          - inventory
          - stockout
          - exception
          - freshglow
        ---

        # FreshGlow Inventory Exception Report

        ## Exception summary

        SKU `FG-DW-LEM-500`, FreshGlow Lemon Dishwash 500 ml, recorded repeated
        inventory exceptions in Karnataka during Q2 2025.

        The affected distributor was Sunrise Distributors.

        ## Stockout history

        | Month | Stockout days | Status |
        |---|---:|---|
        | April 2025 | 8 | Critical |
        | May 2025 | 11 | Critical |
        | June 2025 | 9 | Critical |

        ## Business impact

        The stockouts occurred during the Sparkle Summer 2025 campaign.

        The campaign had planned for a 15% sales lift, but the actual result in
        Karnataka General Trade was approximately 6.8%.

        Product availability was identified as a major constraint.

        ## Root causes

        - Replenishment quantities were below campaign demand.
        - Inventory forecasts did not include the full campaign uplift.
        - Warehouse dispatches were delayed.
        - Distributor inventory reports were not updated frequently.
        - Safety-stock levels were insufficient.

        ## Recommended resolution

        - Increase safety stock by 20% for campaign periods.
        - Produce weekly demand forecasts.
        - Introduce automatic inventory alerts.
        - Review distributor stock before approving campaign activation.
        """
    ).strip(),

    "freshglow-product-launch.md": dedent(
        """
        ---
        document_id: DOC-PRODUCT-001
        document_type: product_launch
        title: FreshGlow Lemon Dishwash Product Launch Note
        brand: FreshGlow
        product: FreshGlow Lemon Dishwash
        sku_ids:
          - FG-DW-LEM-500
          - FG-DW-LEM-1L
        category: Home Care
        effective_date: 2023-06-01
        tags:
          - product-launch
          - freshglow
          - dishwash
        ---

        # FreshGlow Lemon Dishwash Product Launch Note

        ## Product overview

        FreshGlow Lemon Dishwash is a Home Care product designed for everyday
        dish-cleaning requirements.

        The product was launched in two pack sizes:

        | SKU | Pack size |
        |---|---|
        | FG-DW-LEM-500 | 500 ml |
        | FG-DW-LEM-1L | 1 litre |

        ## Target customers

        The product targets value-conscious urban and semi-urban households.

        ## Sales channels

        FreshGlow Lemon Dishwash is available through:

        - General Trade
        - Modern Trade
        - E-commerce

        ## Product positioning

        The key product claims are:

        - Lemon fragrance
        - Grease-removal performance
        - Suitable for daily use
        - Available in value and family pack sizes

        ## Commercial note

        The 500 ml pack is expected to generate higher unit volumes.

        The one-litre pack is expected to provide stronger value perception and
        may receive demand during promotional periods.
        """
    ).strip(),

    "pricing-policy-2025.md": dedent(
        """
        ---
        document_id: DOC-POLICY-001
        document_type: pricing_policy
        title: 2025 Trade Promotion and Pricing Policy
        effective_date: 2025-01-01
        applicable_categories:
          - Home Care
          - Personal Care
          - Packaged Foods
        tags:
          - pricing
          - discounts
          - promotion
          - margin
        ---

        # 2025 Trade Promotion and Pricing Policy

        ## Purpose

        This policy defines discount and promotional controls for Nova Consumer
        Products.

        ## Channel guidance

        Standard discount guidance is:

        | Channel | Standard discount range |
        |---|---:|
        | General Trade | 6% to 10% |
        | Modern Trade | 10% to 14% |
        | E-commerce | 12% to 18% |

        Discounts above the standard range require commercial approval.

        ## Campaign controls

        Campaign proposals must include:

        - Planned sales lift
        - Promotion spend
        - Expected gross margin
        - Inventory-readiness confirmation
        - Distributor execution plan
        - Measurement approach

        ## Margin protection

        Revenue growth must not be treated as successful when it is achieved
        through discounts that create an unacceptable decline in gross-margin
        percentage.

        Regional and channel-level margin must be reviewed alongside revenue
        and unit-sales performance.

        ## Inventory requirement

        Campaign activation should occur only after confirming that sufficient
        inventory is available for the planned sales lift.
        """
    ).strip(),

    "product-discontinuation-notice.md": dedent(
        """
        ---
        document_id: DOC-PRODUCT-002
        document_type: product_discontinuation
        title: PureHome Rose Handwash Discontinuation Notice
        brand: PureHome
        product: PureHome Rose Handwash
        sku_id: PH-HW-ROS-250
        category: Personal Care
        effective_date: 2025-03-31
        tags:
          - discontinuation
          - purehome
          - product-status
        ---

        # PureHome Rose Handwash Discontinuation Notice

        ## Product

        The following product is being discontinued:

        | SKU | Product | Pack size |
        |---|---|---|
        | PH-HW-ROS-250 | PureHome Rose Handwash | 250 ml |

        ## Effective date

        The product will be discontinued effective 31 March 2025.

        ## Reason

        The decision was based on:

        - Low sales volume
        - Product overlap with PureHome Aloe Handwash
        - Increasing manufacturing complexity
        - Limited retailer demand

        ## Historical-document guidance

        Historical reports may continue to mention SKU `PH-HW-ROS-250`.

        The product should not be treated as active for periods after
        31 March 2025.

        ## Replacement recommendation

        Customers and retailers should be directed to:

        - PureHome Aloe Handwash 250 ml
        - SKU `PH-HW-ALO-250`
        """
    ).strip(),
}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_directory = project_root / "data" / "documents"
    output_directory.mkdir(parents=True, exist_ok=True)

    for filename, content in DOCUMENTS.items():
        output_path = output_directory / filename

        output_path.write_text(
            content + "\n",
            encoding="utf-8",
        )

        print(f"Created data/documents/{filename}")

    print(
        f"\nGenerated {len(DOCUMENTS)} overlapping FMCG documents."
    )


if __name__ == "__main__":
    main()