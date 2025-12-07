data_cleaning.py

A full production-grade cleaning pipeline that:

Loads Raw Data

Automatically detects correct delimiter

Reads large files efficiently

Validates expected columns

Cleans Numeric Fields

Converts numeric strings safely

Handles coercion errors

Fixes negatives and unrealistic mmcodes

Cleans Dates

Converts transactionmonth to datetime

Handles invalid formats

Cleans Categorical Fields

Trims whitespace

Replaces empty strings with NaN

Standardizes gender labels ("M", "F")

Handles Missing Values

Drops rows missing essential fields

Fills categorical NAs with "Unknown"

Fills numeric NAs with medians

Feature Engineering

Creates:

lossratio

has_claim

Removes zero-premium rows

Caps loss ratio outliers

Outlier Handling

Capping at 99th percentile for:

TotalPremium

TotalClaims

SumInsured

Saves Clean Output

Writes cleaned dataset to:

      data/interim/cleaned.csv