# Southern California Logistics Operations
## Standard Operating Procedure (SOP): Cold-Chain & Transit Anomalies
**Version:** 3.0 | **Effective Date:** Jan 2026 | **Confidentiality:** Internal Operations Only

### 1. Temperature Control & Spoilage Prevention (Fresh Perishables)
All refrigerated fleets carrying fresh perishables must maintain strict IoT temperature compliance to prevent cargo spoilage.
* **Fresh Perishables:** IoT temperature must remain between **0.0°C and 4.0°C**.
* **Critical Breach:** If `IOT_TEMP_VAL_C` exceeds **4.0°C**, an immediate cold-chain breach is declared.
* **Mitigation Protocol:** The dispatcher must immediately contact the driver to restart the auxiliary cooling unit. If ETA delay is greater than 1 hour, divert the vehicle to the nearest emergency cold-storage facility.

### 2. Frozen Cargo & Deep-Freeze Compliance
* **Frozen Perishables / Meat / Seafood:** IoT temperature must remain at or below **-18.0°C**.
* **Thaw Warning:** Any reading above **-15.0°C** triggers a Tier 1 compressor inspection and emergency dry-ice/cold-docking contingency.

### 3. Biologics & Pharmaceutical Cold-Chain
* **Pharma / Vaccines:** IoT temperature must remain strictly between **2.0°C and 8.0°C** with zero tolerance for excursions exceeding 30 minutes.
* **Escalation:** Any pharma excursion outside **2.0°C–8.0°C** requires immediate **Tier 2** Quality Assurance & Logistics Manager escalation.

### 4. Route Congestion & Diversion Tactics
Port congestion heavily impacts SLA compliance.
* **Port of Long Beach / LA:** If port congestion level (`PRT_CNG_LVL`) exceeds a severity index of **7.0**, standard routing is suspended.
* **Mitigation Protocol:** Do not hold freight at the port. Divert all active shipments to the **Inland Empire Overflow Depot (San Bernardino)** for cross-docking.

### 5. Risk Classification & Escalation Triggers
* **Tier 1 Escalation:** Single-metric warning (e.g., moderate weather caution or minor delay probability between 0.40 and 0.65) handled by shift dispatcher.
* **Tier 2 Escalation:** Any shipment classified as **"High Risk"** (`RISK_CLS_TXT` = High Risk) combined with a delay probability (`DELAY_PROB_DEC`) greater than **0.65** must be escalated to the Tier 2 Logistics Manager.
