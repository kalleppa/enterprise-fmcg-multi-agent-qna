from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
START_MONTH = "2024-01-01"
END_MONTH = "2025-06-01"


def build_products() -> pd.DataFrame:
    """Create the FMCG product master."""

    rows = [
        (
            "FG-DW-LEM-500",
            "FreshGlow Lemon Dishwash",
            "FreshGlow",
            "Home Care",
            "Dishwash",
            "500 ml",
            "2023-06-01",
            None,
            "Active",
            115,
            0.54,
        ),
        (
            "FG-DW-LEM-1L",
            "FreshGlow Lemon Dishwash",
            "FreshGlow",
            "Home Care",
            "Dishwash",
            "1 litre",
            "2023-06-01",
            None,
            "Active",
            205,
            0.54,
        ),
        (
            "FG-SC-LAV-500",
            "FreshGlow Lavender Surface Cleaner",
            "FreshGlow",
            "Home Care",
            "Surface Cleaner",
            "500 ml",
            "2023-09-01",
            None,
            "Active",
            135,
            0.56,
        ),
        (
            "FG-SC-LAV-1L",
            "FreshGlow Lavender Surface Cleaner",
            "FreshGlow",
            "Home Care",
            "Surface Cleaner",
            "1 litre",
            "2023-09-01",
            None,
            "Active",
            235,
            0.56,
        ),
        (
            "PH-HW-ALO-250",
            "PureHome Aloe Handwash",
            "PureHome",
            "Personal Care",
            "Handwash",
            "250 ml",
            "2023-04-01",
            None,
            "Active",
            95,
            0.50,
        ),
        (
            "PH-HW-ALO-500",
            "PureHome Aloe Handwash",
            "PureHome",
            "Personal Care",
            "Handwash",
            "500 ml",
            "2023-04-01",
            None,
            "Active",
            165,
            0.50,
        ),
        (
            "PH-BW-NEE-500",
            "PureHome Neem Body Wash",
            "PureHome",
            "Personal Care",
            "Body Wash",
            "500 ml",
            "2023-11-01",
            None,
            "Active",
            225,
            0.52,
        ),
        (
            "PH-HW-ROS-250",
            "PureHome Rose Handwash",
            "PureHome",
            "Personal Care",
            "Handwash",
            "250 ml",
            "2022-08-01",
            "2025-03-31",
            "Discontinued",
            90,
            0.51,
        ),
        (
            "NB-GR-CHO-250",
            "NutriBite Chocolate Granola",
            "NutriBite",
            "Packaged Foods",
            "Breakfast Cereal",
            "250 g",
            "2023-02-01",
            None,
            "Active",
            180,
            0.61,
        ),
        (
            "NB-GR-HON-500",
            "NutriBite Honey Granola",
            "NutriBite",
            "Packaged Foods",
            "Breakfast Cereal",
            "500 g",
            "2023-02-01",
            None,
            "Active",
            320,
            0.61,
        ),
        (
            "NB-PB-CHO-6P",
            "NutriBite Chocolate Protein Bars",
            "NutriBite",
            "Packaged Foods",
            "Protein Bars",
            "Pack of 6",
            "2024-01-01",
            None,
            "Active",
            270,
            0.60,
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "sku_id",
            "product_name",
            "brand",
            "category",
            "subcategory",
            "pack_size",
            "launch_date",
            "discontinuation_date",
            "status",
            "unit_price_inr",
            "base_cogs_ratio",
        ],
    )


def build_geography() -> pd.DataFrame:
    """Create the geography hierarchy."""

    rows = [
        ("GEO-S-001", "India", "South Region", "Karnataka", "Bengaluru"),
        ("GEO-S-002", "India", "South Region", "Tamil Nadu", "Chennai"),
        ("GEO-W-001", "India", "West Region", "Maharashtra", "Mumbai"),
        ("GEO-W-002", "India", "West Region", "Gujarat", "Ahmedabad"),
        ("GEO-N-001", "India", "North Region", "Delhi", "Delhi"),
        ("GEO-N-002", "India", "North Region", "Uttar Pradesh", "Lucknow"),
        ("GEO-E-001", "India", "East Region", "West Bengal", "Kolkata"),
        ("GEO-E-002", "India", "East Region", "Odisha", "Bhubaneswar"),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "geography_id",
            "country",
            "region",
            "state",
            "city",
        ],
    )


def build_distributors() -> pd.DataFrame:
    """Create distributors for each region and channel."""

    rows = [
        (
            "DST-S-GT-01",
            "Sunrise Distributors",
            "South Region",
            "Karnataka",
            "General Trade",
            True,
        ),
        (
            "DST-S-MT-01",
            "Southern Retail Supply",
            "South Region",
            "Tamil Nadu",
            "Modern Trade",
            True,
        ),
        (
            "DST-S-EC-01",
            "South Digital Fulfilment",
            "South Region",
            "Karnataka",
            "E-commerce",
            True,
        ),
        (
            "DST-W-GT-01",
            "Western Trade Partners",
            "West Region",
            "Maharashtra",
            "General Trade",
            True,
        ),
        (
            "DST-W-MT-01",
            "West Metro Retail Supply",
            "West Region",
            "Gujarat",
            "Modern Trade",
            True,
        ),
        (
            "DST-W-EC-01",
            "West Digital Fulfilment",
            "West Region",
            "Maharashtra",
            "E-commerce",
            True,
        ),
        (
            "DST-N-GT-01",
            "Capital Distribution Services",
            "North Region",
            "Delhi",
            "General Trade",
            True,
        ),
        (
            "DST-N-MT-01",
            "North Star Retail Supply",
            "North Region",
            "Uttar Pradesh",
            "Modern Trade",
            True,
        ),
        (
            "DST-N-EC-01",
            "North Digital Fulfilment",
            "North Region",
            "Delhi",
            "E-commerce",
            True,
        ),
        (
            "DST-E-GT-01",
            "Eastern Consumer Supply",
            "East Region",
            "West Bengal",
            "General Trade",
            True,
        ),
        (
            "DST-E-MT-01",
            "East Metro Retail Supply",
            "East Region",
            "Odisha",
            "Modern Trade",
            True,
        ),
        (
            "DST-E-EC-01",
            "East Digital Fulfilment",
            "East Region",
            "West Bengal",
            "E-commerce",
            True,
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "distributor_id",
            "distributor_name",
            "region",
            "state",
            "channel",
            "active_status",
        ],
    )


def build_sales(
    products: pd.DataFrame,
    geography: pd.DataFrame,
    distributors: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate monthly sales data with controlled business scenarios."""

    months = pd.date_range(START_MONTH, END_MONTH, freq="MS")

    product_volume = {
        "FG-DW-LEM-500": 1100,
        "FG-DW-LEM-1L": 720,
        "FG-SC-LAV-500": 850,
        "FG-SC-LAV-1L": 560,
        "PH-HW-ALO-250": 950,
        "PH-HW-ALO-500": 650,
        "PH-BW-NEE-500": 480,
        "PH-HW-ROS-250": 350,
        "NB-GR-CHO-250": 700,
        "NB-GR-HON-500": 520,
        "NB-PB-CHO-6P": 430,
    }

    region_factor = {
        "South Region": 1.16,
        "West Region": 1.08,
        "North Region": 0.98,
        "East Region": 0.86,
    }

    channel_factor = {
        "General Trade": 1.00,
        "Modern Trade": 0.74,
        "E-commerce": 0.48,
    }

    distributor_lookup = distributors.set_index(
        ["region", "channel"]
    )["distributor_id"].to_dict()

    product_lookup = products.set_index("sku_id").to_dict("index")

    rows = []

    for month in months:
        for product in products.itertuples(index=False):

            if (
                pd.notna(product.discontinuation_date)
                and month > pd.Timestamp(product.discontinuation_date)
            ):
                continue

            for geo in geography.itertuples(index=False):
                for channel, channel_multiplier in channel_factor.items():

                    units = (
                        product_volume[product.sku_id]
                        * region_factor[geo.region]
                        * channel_multiplier
                    )

                    months_from_start = (
                        (month.year - 2024) * 12 + month.month - 1
                    )

                    units *= 1 + months_from_start * 0.006

                    seasonal_factor = 1 + 0.06 * np.sin(
                        (month.month - 1) / 12 * 2 * np.pi
                    )
                    units *= seasonal_factor

                    # Successful NutriBite campaign.
                    if (
                        product.sku_id == "NB-GR-HON-500"
                        and geo.region == "West Region"
                        and channel == "E-commerce"
                        and pd.Timestamp("2025-01-01")
                        <= month
                        <= pd.Timestamp("2025-03-01")
                    ):
                        units *= 1.22

                    # Underperforming Sparkle Summer campaign.
                    if (
                        product.sku_id == "FG-DW-LEM-500"
                        and geo.region == "South Region"
                        and channel in {"General Trade", "Modern Trade"}
                        and pd.Timestamp("2025-04-01")
                        <= month
                        <= pd.Timestamp("2025-06-01")
                    ):
                        units *= 1.07

                    # Sunrise Distributors underperformance.
                    if (
                        geo.region == "South Region"
                        and channel == "General Trade"
                        and pd.Timestamp("2025-01-01")
                        <= month
                        <= pd.Timestamp("2025-06-01")
                    ):
                        units *= 0.88

                    units_sold = max(
                        0,
                        int(round(units * rng.normal(1.0, 0.04))),
                    )

                    product_data = product_lookup[product.sku_id]
                    unit_price = float(product_data["unit_price_inr"])

                    gross_revenue = round(
                        units_sold * unit_price,
                        2,
                    )

                    discount_pct = {
                        "General Trade": 0.08,
                        "Modern Trade": 0.12,
                        "E-commerce": 0.15,
                    }[channel]

                    # West Region has higher sales but lower margin in 2025.
                    if geo.region == "West Region" and month.year == 2025:
                        discount_pct += 0.05

                    if (
                        product.sku_id == "FG-DW-LEM-500"
                        and geo.region == "South Region"
                        and pd.Timestamp("2025-04-01")
                        <= month
                        <= pd.Timestamp("2025-06-01")
                    ):
                        discount_pct += 0.03

                    discount = round(
                        gross_revenue * discount_pct,
                        2,
                    )

                    net_revenue = round(
                        gross_revenue - discount,
                        2,
                    )

                    cogs_ratio = float(
                        product_data["base_cogs_ratio"]
                    )

                    cogs = round(
                        gross_revenue * cogs_ratio,
                        2,
                    )

                    gross_margin = round(
                        net_revenue - cogs,
                        2,
                    )

                    gross_margin_pct = round(
                        (
                            gross_margin
                            / net_revenue
                            * 100
                        )
                        if net_revenue
                        else 0,
                        2,
                    )

                    rows.append(
                        {
                            "month": month.date().isoformat(),
                            "sku_id": product.sku_id,
                            "geography_id": geo.geography_id,
                            "distributor_id": distributor_lookup[
                                (geo.region, channel)
                            ],
                            "channel": channel,
                            "units_sold": units_sold,
                            "gross_revenue_inr": gross_revenue,
                            "discount_inr": discount,
                            "net_revenue_inr": net_revenue,
                            "cogs_inr": cogs,
                            "gross_margin_inr": gross_margin,
                            "gross_margin_pct": gross_margin_pct,
                        }
                    )

    return pd.DataFrame(rows)


def build_inventory(
    sales: pd.DataFrame,
    geography: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create inventory data using the generated sales data."""

    geography_lookup = geography.set_index(
        "geography_id"
    ).to_dict("index")

    rows = []

    for sale in sales.itertuples(index=False):
        geo = geography_lookup[sale.geography_id]
        month = pd.Timestamp(sale.month)

        opening_stock = max(
            0,
            int(round(sale.units_sold * rng.uniform(0.20, 0.35))),
        )

        received_units = max(
            0,
            int(round(sale.units_sold * rng.uniform(0.82, 1.05))),
        )

        available_units = opening_stock + received_units

        closing_stock = max(
            0,
            available_units - sale.units_sold,
        )

        if available_units < sale.units_sold:
            stockout_days = int(rng.integers(2, 6))
        else:
            stockout_days = int(rng.integers(0, 2))

        # Deliberate Sparkle Summer stockout scenario.
        if (
            sale.sku_id == "FG-DW-LEM-500"
            and geo["state"] == "Karnataka"
            and sale.channel == "General Trade"
            and pd.Timestamp("2025-04-01")
            <= month
            <= pd.Timestamp("2025-06-01")
        ):
            opening_stock = int(
                round(sale.units_sold * 0.10)
            )

            received_units = int(
                round(sale.units_sold * 0.72)
            )

            closing_stock = max(
                0,
                opening_stock
                + received_units
                - sale.units_sold,
            )

            stockout_days = {
                4: 8,
                5: 11,
                6: 9,
            }[month.month]

        if stockout_days >= 7:
            inventory_status = "Critical"
        elif stockout_days >= 3:
            inventory_status = "At Risk"
        else:
            inventory_status = "Healthy"

        rows.append(
            {
                "month": sale.month,
                "sku_id": sale.sku_id,
                "geography_id": sale.geography_id,
                "distributor_id": sale.distributor_id,
                "channel": sale.channel,
                "opening_stock_units": opening_stock,
                "received_units": received_units,
                "units_sold": sale.units_sold,
                "closing_stock_units": closing_stock,
                "stockout_days": stockout_days,
                "inventory_status": inventory_status,
            }
        )

    return pd.DataFrame(rows)


def build_promotions() -> pd.DataFrame:
    """Create promotion data containing known outcomes."""

    rows = [
        (
            "CMP-2025-001",
            "Sparkle Summer 2025",
            "FG-DW-LEM-500",
            "GEO-S-001",
            "General Trade",
            "2025-04-01",
            "2025-06-30",
            2_500_000,
            15.0,
            6.8,
            "Completed",
        ),
        (
            "CMP-2025-001",
            "Sparkle Summer 2025",
            "FG-DW-LEM-500",
            "GEO-S-002",
            "Modern Trade",
            "2025-04-01",
            "2025-06-30",
            1_800_000,
            15.0,
            7.4,
            "Completed",
        ),
        (
            "CMP-2025-002",
            "NutriBite Digital Boost",
            "NB-GR-HON-500",
            "GEO-W-001",
            "E-commerce",
            "2025-01-01",
            "2025-03-31",
            1_200_000,
            18.0,
            22.1,
            "Completed",
        ),
        (
            "CMP-2025-002",
            "NutriBite Digital Boost",
            "NB-GR-HON-500",
            "GEO-W-002",
            "E-commerce",
            "2025-01-01",
            "2025-03-31",
            800_000,
            18.0,
            20.6,
            "Completed",
        ),
        (
            "CMP-2024-003",
            "PureHome Hygiene Week",
            "PH-HW-ALO-250",
            "GEO-N-001",
            "Modern Trade",
            "2024-08-01",
            "2024-08-31",
            650_000,
            10.0,
            11.3,
            "Completed",
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "campaign_id",
            "campaign_name",
            "sku_id",
            "geography_id",
            "channel",
            "start_date",
            "end_date",
            "promotion_spend_inr",
            "planned_sales_lift_pct",
            "actual_sales_lift_pct",
            "campaign_status",
        ],
    )


def validate_data(
    products: pd.DataFrame,
    geography: pd.DataFrame,
    distributors: pd.DataFrame,
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    promotions: pd.DataFrame,
) -> None:
    """Validate foreign keys and financial calculations."""

    product_ids = set(products["sku_id"])
    geography_ids = set(geography["geography_id"])
    distributor_ids = set(distributors["distributor_id"])

    assert set(sales["sku_id"]).issubset(product_ids)
    assert set(inventory["sku_id"]).issubset(product_ids)
    assert set(promotions["sku_id"]).issubset(product_ids)

    assert set(sales["geography_id"]).issubset(geography_ids)
    assert set(inventory["geography_id"]).issubset(
        geography_ids
    )
    assert set(promotions["geography_id"]).issubset(
        geography_ids
    )

    assert set(sales["distributor_id"]).issubset(
        distributor_ids
    )
    assert set(inventory["distributor_id"]).issubset(
        distributor_ids
    )

    net_revenue_difference = (
        sales["gross_revenue_inr"]
        - sales["discount_inr"]
        - sales["net_revenue_inr"]
    ).abs()

    assert (net_revenue_difference < 0.01).all()

    gross_margin_difference = (
        sales["net_revenue_inr"]
        - sales["cogs_inr"]
        - sales["gross_margin_inr"]
    ).abs()

    assert (gross_margin_difference < 0.01).all()

    assert (
        inventory["closing_stock_units"] >= 0
    ).all()


def main() -> None:
    rng = np.random.default_rng(SEED)

    project_root = Path(__file__).resolve().parents[1]
    output_directory = project_root / "data" / "structured"
    output_directory.mkdir(parents=True, exist_ok=True)

    products = build_products()
    geography = build_geography()
    distributors = build_distributors()

    sales = build_sales(
        products,
        geography,
        distributors,
        rng,
    )

    inventory = build_inventory(
        sales,
        geography,
        rng,
    )

    promotions = build_promotions()

    validate_data(
        products,
        geography,
        distributors,
        sales,
        inventory,
        promotions,
    )

    datasets = {
        "products.csv": products,
        "geography.csv": geography,
        "distributors.csv": distributors,
        "sales.csv": sales,
        "inventory.csv": inventory,
        "promotions.csv": promotions,
    }

    for filename, dataframe in datasets.items():
        output_path = output_directory / filename

        dataframe.to_csv(
            output_path,
            index=False,
        )

        print(
            f"Created data/structured/{filename}: "
            f"{len(dataframe):,} rows"
        )

    print("\nSynthetic FMCG datasets generated successfully.")


if __name__ == "__main__":
    main()