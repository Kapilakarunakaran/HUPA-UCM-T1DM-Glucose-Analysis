# 12_Python-Pioneers_Python-Hacathon_MAY-2026
hupa-ucm-t1dm-glucose-analysis
End-to-end data analysis pipeline for Continuous Glucose Monitoring (CGM)
data from 25 people with Type 1 Diabetes Mellitus (T1DM).

**Dataset:** HUPA-UCM (Mendeley Data)
DOI: 10.17632/3hbcscwz44.1 | License: CC BY 4.0
Published: April 2024 | Hospital Universitario Principe de Asturias
& Universidad Complutense de Madrid, Spain

# HUPA-UCM T1DM Glucose Analysis

End-to-end data analysis pipeline for Continuous Glucose Monitoring (CGM)
data from 25 people with Type 1 Diabetes Mellitus (T1DM).

**Dataset:** HUPA-UCM (Mendeley Data)
DOI: 10.17632/3hbcscwz44.1 | License: CC BY 4.0
Published: April 2024 | Hospital Universitario Principe de Asturias
& Universidad Complutense de Madrid, Spain


## What This Project Does

- Combines 25 individual patient CSV files into one normalized master dataset
- Applies a 14-day window normalization strategy to handle patients with
  unequal observation periods (7 days to 574 days)
- Engineers clinical features: Time in Range (TIR), glucose zones,
  meal/bolus flags, time-of-day periods, relative day numbers
- Joins demographics (age, gender, race, sleep quality) to time-series data
- Performs exploratory analysis across glucose patterns, sleep correlation,
  and demographic comparisons

---

## Dataset Summary

| Property          | Value                          |
|-------------------|-------------------------------|
| Patients          | 25 with T1DM                  |
| Total Rows        | 89,615                        |
| Columns           | 21                            |
| Null Values       | 0                             |
| Sampling Interval | Every 5 minutes               |
| CGM Device        | FreeStyle Libre 2             |
| Activity Device   | Fitbit Ionic                  |
| Observation Window| Normalized to first 14 days   |

---

## Variables

| Column                  | Description                                      |
|-------------------------|--------------------------------------------------|
| patient_id              | Unique patient identifier                        |
| time                    | Timestamp (5-min intervals)                      |
| day_num                 | Day number relative to patient start (1-14)      |
| glucose                 | Blood glucose in mg/dL                           |
| calories                | Calories burned (Fitbit)                         |
| heart_rate              | Heart rate in BPM (Fitbit)                       |
| steps                   | Step count per 5-min interval                    |
| basal_rate              | Basal insulin infusion (units/hr)                |
| bolus_volume_delivered  | Bolus insulin delivered (units)                  |
| carb_input              | Carbohydrates ingested (grams)                   |
| Age, Gender, Race       | Patient demographics                             |
| Sleep Duration/Quality  | Fitbit sleep tracking metrics                    |
| glucose_zone            | Clinical band (Critical Low to Critical High)    |
| time_of_day             | Night / Morning / Afternoon / Evening            |
| is_meal / is_bolus      | Binary event flags                               |

---

## Glucose Zone Reference

| Zone          | Range (mg/dL) | Clinical Meaning              |
|---------------|---------------|-------------------------------|
| Critical Low  | < 54          | Severe hypoglycemia           |
| Low           | 54 – 70       | Hypoglycemia                  |
| In Range      | 70 – 180      | Target (goal: ≥ 70% TIR)     |
| High          | 180 – 250     | Hyperglycemia                 |
| Critical High | > 250         | Severe hyperglycemia          |

---

## Key Results

| Metric               | Value    |
|----------------------|----------|
| Overall Mean Glucose | 153.1 mg/dL |
| Overall TIR          | 61.4%    |
| Hypoglycemia Rate    | 7.7%     |
| Hyperglycemia Rate   | 30.7%    |
| Best TIR (patient)   | 93.4% — HUPA0028P |
| Worst TIR (patient)  | 35.6% — HUPA0017P |
