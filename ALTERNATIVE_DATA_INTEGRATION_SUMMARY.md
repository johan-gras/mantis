# Alternative Data Integration Summary for Mantis Backtesting Framework

**Research Date**: 2026-01-15
**Framework**: Mantis (formerly Ralph-Rust-Backtest)

## Executive Summary

Alternative data has become central to modern quantitative trading, with the global market reaching $7.2 billion in 2023 and growing at 50% annually. Investment firms now spend approximately $900,000 per year on alternative data sources. This document summarizes the key alternative data types and integration requirements for a production-grade backtesting framework.

---

## 1. News and Sentiment Data

### Overview
Sentiment analysis from news sources, social media, and financial reports has become a critical alpha signal, with recent backtesting studies showing returns of 50.63% over 28 months for sentiment-based strategies on Dow Jones 30 stocks.

### Data Sources
- **Commercial Providers**: Bloomberg, Reuters, RavenPack, Refinitiv
- **Social Media**: Twitter/X, Reddit (WallStreetBets), StockTwits
- **News Aggregators**: Seeking Alpha, Financial Times, WSJ
- **Alternative Sources**: Earnings call transcripts, SEC filings (8-K, 10-K)

### Integration Requirements
1. **Real-time Sentiment Scoring**
   - NLP pipeline for text analysis
   - Sentiment polarity (positive/negative/neutral)
   - Entity extraction (company, ticker, industry)
   - Confidence scores for each classification

2. **Backtesting Considerations**
   - Point-in-time data alignment (news published timestamp vs market timestamp)
   - Handling news released outside trading hours
   - Event study windows (pre-announcement, post-announcement)
   - Sentiment decay models (how long does news impact last?)

3. **Data Format**
   ```rust
   pub struct SentimentEvent {
       pub timestamp: DateTime<Utc>,
       pub symbol: String,
       pub source: String,           // "bloomberg", "twitter", "earnings_call"
       pub headline: String,
       pub sentiment_score: f64,     // -1.0 to 1.0
       pub confidence: f64,           // 0.0 to 1.0
       pub entities: Vec<String>,
       pub event_type: EventType,    // "earnings", "product_launch", "regulatory"
   }
   ```

4. **Framework Integration Points**
   - Add sentiment as feature in `features.rs`
   - Sentiment-based signal generation in strategy
   - Preprocessing for ML models (embeddings, TF-IDF)

### Backtesting Tools & Platforms
- **SentimenTrader**: 16,000+ indicators with proprietary backtesting engine
- **Python Libraries**: backtrader + sentiment feeds, specialized GenAI integrations
- **Binance AI Backtester**: Reads headline sentiment at each candlestick interval

---

## 2. Satellite Imagery

### Overview
Satellite imagery provides unparalleled, unbiased intelligence for investment decisions, with hedge funds achieving 4-5% informational advantage in the three days around quarterly earnings announcements using parking lot analysis.

### Use Cases by Sector

#### Retail (High Priority)
- **Parking lot vehicle counts**: Year-over-year changes predict quarterly sales
- **Construction activity**: New store openings, expansion projects
- **Foot traffic patterns**: Consumer behavior analysis
- **Provider Example**: RS Metrics (pioneered parking lot analysis in 2011)

#### Energy & Commodities
- **Oil storage tanks**: Capacity tracking, inventory levels
- **Tanker movements**: Supply chain visibility
- **Flaring activity**: Production levels at oil fields
- **Pipeline monitoring**: Flow rates, infrastructure changes

#### Agriculture
- **Crop health monitoring**: NDVI (Normalized Difference Vegetation Index)
- **Yield predictions**: Pre-harvest estimates
- **Drought detection**: Water stress indicators
- **Planting/harvesting timelines**: Seasonal signals

#### Mining & Materials
- **Stockpile volumes**: Inventory at mines/ports
- **Equipment activity**: Production indicators
- **Infrastructure development**: Capacity expansion signals

### Data Providers
- **Orbital Insight**: AI-powered satellite analysis
- **SpaceKnow**: Computer vision on satellite imagery
- **Planet Labs**: Daily high-resolution imagery
- **PierSight**: SAR + AIS for maritime surveillance
- **Maxar/DigitalGlobe**: High-resolution commercial imagery

### Integration Requirements

1. **Data Types**
   ```rust
   pub struct SatelliteSignal {
       pub timestamp: DateTime<Utc>,
       pub symbol: String,
       pub location: GeoLocation,       // lat/lon of observed site
       pub metric_type: MetricType,     // "vehicle_count", "tank_capacity", "ndvi"
       pub value: f64,
       pub baseline: f64,               // historical average
       pub percent_change: f64,
       pub confidence: f64,
       pub image_date: DateTime<Utc>,   // actual image capture time
   }

   pub enum MetricType {
       VehicleCount,
       TankCapacity,
       NDVI,
       ConstructionProgress,
       ShipCount,
   }
   ```

2. **Computer Vision Pipeline**
   - Object detection models (R-CNN, YOLO v8, U-Net, DeepLab)
   - Change detection algorithms
   - Time-series analysis of satellite metrics
   - Aggregation from multiple locations to company-level signals

3. **Backtesting Challenges**
   - **Temporal lag**: Images captured days before processing
   - **Cloud cover**: Missing data periods
   - **Sampling frequency**: Daily vs weekly imagery
   - **Geographic aggregation**: Multiple store locations → single ticker signal

4. **Framework Integration**
   - New data loader for satellite time-series
   - Geographic aggregation utilities
   - Lag adjustment for image processing delays
   - Confidence-weighted signal blending

---

## 3. Web Scraping

### Overview
Web scraping remains the #1 alternative data source, with the global market hitting $7.2 billion. Quant teams use scraped data to detect trend shifts in fundamentals, sentiment, and price reactions.

### High-Value Scraping Targets

#### E-commerce & Pricing
- **Product prices**: Track competitor pricing strategies
- **Price elasticity**: Demand response to price changes
- **Inventory availability**: Stock-outs as demand signals
- **Customer reviews**: Sentiment, quality issues
- **Seller ratings**: Platform health (Amazon, eBay)

#### Corporate Activity
- **Job postings**: Hiring trends (growth signals)
- **LinkedIn data**: Headcount changes, employee sentiment
- **Glassdoor reviews**: Employee satisfaction, turnover risk
- **Patent filings**: Innovation indicators

#### Supply Chain
- **Shipping manifests**: Import/export volumes
- **Port activity**: Container tracking
- **Supplier relationships**: Dependency analysis

#### Real Estate & Construction
- **Building permits**: Development pipeline
- **Property listings**: Market dynamics
- **Construction starts**: Economic activity

### Data Providers
- **Bright Data**: Comprehensive scraping infrastructure
- **Import Genius**: Shipping records, customs data
- **Trademo**: 164M+ cargo records with ML/NLP processing
- **PromptCloud**: Finance-focused web scraping

### Integration Requirements

1. **Data Schema**
   ```rust
   pub struct WebScrapedData {
       pub timestamp: DateTime<Utc>,
       pub symbol: String,
       pub data_type: ScrapedDataType,
       pub source_url: String,
       pub metric: String,              // "job_postings", "avg_price", "review_score"
       pub value: f64,
       pub raw_data: Option<String>,    // JSON blob for flexibility
       pub scrape_date: DateTime<Utc>,  // when scraped (may differ from publication)
   }

   pub enum ScrapedDataType {
       JobPosting,
       ProductPrice,
       Review,
       Inventory,
       SupplyChain,
   }
   ```

2. **Data Quality Issues**
   - **Website changes**: Scraper breakage
   - **Rate limiting**: Gaps in data
   - **Duplicates**: Same data scraped multiple times
   - **Stale data**: Cached vs fresh
   - **Normalization**: Different units, currencies, formats

3. **Alpha Signal Examples**
   - Job posting increases → revenue growth correlation
   - Product price cuts → margin contraction
   - Review sentiment spikes → volatility expansion
   - Inventory depletion → demand surge

4. **Framework Integration**
   - Flexible JSON parsing for varied schemas
   - Deduplication logic
   - Outlier detection for scraping errors
   - Normalization pipelines

---

## 4. Credit Card Transaction Data

### Overview
Aggregated and anonymized credit card data provides a "living, breathing map of the economy," giving investors unparalleled visibility into revenue flows before companies report earnings.

### Use Cases

#### Revenue Forecasting
- **Real-time sales tracking**: Daily/weekly revenue estimates
- **Geographic breakdowns**: Regional performance
- **Customer segmentation**: Demographics, spending patterns
- **Category performance**: Product mix shifts

#### Risk Management
- **Credit underwriting**: Transaction trends for lending decisions
- **Portfolio monitoring**: Customer health metrics
- **Fraud detection**: Unusual patterns

#### Consumer Behavior Analysis
- **Spending seasonality**: Holiday effects, cycles
- **Competitive dynamics**: Wallet share shifts
- **Brand switching**: Customer migration patterns

### Data Providers
- **Facteus**: Debit/credit card transaction data
- **Plaid**: Transaction API, bank account history
- **Envestnet Yodlee**: Financial data aggregation
- **Enigma**: Card transaction datasets

### Data Attributes
- Transaction amounts, timestamps, merchant categories
- Consumer demographics (age, income, location)
- Merchant-level granularity
- Aggregated statistics (anonymized)

### Integration Requirements

1. **Data Schema**
   ```rust
   pub struct TransactionAggregate {
       pub timestamp: DateTime<Utc>,
       pub symbol: String,              // public company ticker
       pub merchant_category: String,
       pub transaction_count: u64,
       pub total_amount: f64,
       pub avg_transaction: f64,
       pub yoy_growth_pct: f64,
       pub geographic_region: Option<String>,
       pub customer_segment: Option<String>,
   }
   ```

2. **Privacy Considerations**
   - Always aggregated (never individual transactions)
   - Anonymized customer data
   - Minimum aggregation thresholds
   - Compliance with data regulations

3. **Backtesting Challenges**
   - **Reporting lag**: Transaction date vs data availability
   - **Sampling bias**: Not all merchants covered
   - **Seasonality**: Strong calendar effects
   - **Normalization**: Adjusting for panel composition changes

4. **Alpha Generation**
   - Lead indicator for quarterly earnings
   - Early warning for sales deterioration
   - Market share tracking vs competitors

---

## 5. Weather Data

### Overview
Weather is no longer just risk—it's a signal for commodity traders. Top hedge funds build weather into their decision stacks because 1-in-20-year events are becoming annual occurrences.

### Trading Applications

#### Energy Markets
- **Natural gas**: Heating/Cooling Degree Days (HDD/CDD)
- **Electricity**: Demand forecasting
- **Renewable energy**: Wind/solar generation predictions

#### Agriculture
- **Crop yields**: Precipitation, temperature stress
- **Planting/harvest timing**: Growing degree days
- **Frost risk**: Damage to crops

#### Retail
- **Seasonal products**: Ice cream, winter clothing demand
- **Foot traffic**: Weather impact on store visits

### Key Metrics
- **Agricultural Climate Index**: Crop risk monitoring
- **Energy Demand Index**: HDD/CDD for gas/electricity
- **Forecast accuracy**: Historical forecast vs actual
- **Extreme events**: Hurricanes, droughts, floods

### Data Providers
- **Commodity Weather Group**: Ag/energy specialization
- **Climavision**: Commodity trading forecasts
- **World Climate Service**: Historical forecast archive
- **OpenWeather**: API access to forecasts
- **Ehab.co WeatherWise**: Finance signals

### Integration Requirements

1. **Data Schema**
   ```rust
   pub struct WeatherSignal {
       pub timestamp: DateTime<Utc>,
       pub symbol: String,              // commodity or affected stock
       pub location: GeoLocation,
       pub metric_type: WeatherMetric,
       pub value: f64,
       pub forecast: Option<f64>,       // predicted value
       pub forecast_error: Option<f64>, // historical accuracy
       pub severity: Option<String>,    // "normal", "extreme"
   }

   pub enum WeatherMetric {
       Temperature,
       Precipitation,
       HDD,                             // Heating Degree Days
       CDD,                             // Cooling Degree Days
       WindSpeed,
       SolarIrradiance,
   }
   ```

2. **Trading Strategies**
   - **Forecast surprises**: Market reactions to unexpected forecasts
   - **Forecast errors**: Trading forecast performance patterns
   - **Forecast momentum**: Continued model changes in same direction

3. **Backtesting Considerations**
   - **Point-in-time forecasts**: Use forecasts available at decision time (not actuals)
   - **Forecast revisions**: Track how predictions change over time
   - **Geographic aggregation**: Multiple locations → commodity signal
   - **Lag handling**: Weather impact timing varies by commodity

---

## 6. Supply Chain Data

### Overview
Supply chain data has become critical for understanding corporate performance, with data vendors providing information to hedge funds trying to anticipate the impact of supply chain issues on their investments.

### Data Types

#### Shipping & Logistics
- **Import/export records**: Customs data, bills of lading
- **Container tracking**: Port activity, transit times
- **Freight rates**: Baltic Dry Index, container shipping costs
- **Vessel movements**: AIS (Automatic Identification System)

#### Supplier Relationships
- **Supplier networks**: Who supplies whom
- **Dependency analysis**: Critical path suppliers
- **Geographic concentration**: Country/region exposure
- **Alternative suppliers**: Redundancy analysis

#### Inventory & Production
- **Just-in-time metrics**: Inventory velocity
- **Lead times**: Order to delivery duration
- **Production schedules**: Factory activity

### Major Providers
- **Panjiva (S&P Global)**: Global trade intelligence, 50%+ of global trade by dollar value
- **Altana**: AI-powered supply chain mapping, billions of data points
- **Import Genius**: Shipping records database
- **Trademo**: 164M+ cargo records with ML/NLP analysis
- **Descartes Datamyne**: Trade data analytics

### Integration Requirements

1. **Data Schema**
   ```rust
   pub struct SupplyChainEvent {
       pub timestamp: DateTime<Utc>,
       pub symbol: String,
       pub event_type: SupplyChainEventType,
       pub supplier: Option<String>,
       pub origin_country: String,
       pub destination_country: String,
       pub cargo_volume: Option<f64>,
       pub cargo_value: Option<f64>,
       pub transit_days: Option<u32>,
       pub disruption_indicator: Option<bool>,
   }

   pub enum SupplyChainEventType {
       Import,
       Export,
       Shipment,
       PortDelay,
       SupplierChange,
       Disruption,
   }
   ```

2. **Use Cases**
   - **Disruption tracking**: Early warning for production issues
   - **Revenue estimation**: Import volumes → sales forecasts
   - **Competitive intelligence**: Supplier changes, market share shifts
   - **Risk assessment**: Geographic concentration, single points of failure

3. **Backtesting Challenges**
   - **Reporting lag**: Customs data released with delay
   - **Partial visibility**: Not all trade captured
   - **Company matching**: Linking entities across datasets
   - **Aggregation complexity**: Multiple shipments → company metric

---

## 7. ESG Data

### Overview
ESG (Environmental, Social, Governance) data integration in quantitative trading has become well-established, with statistically significant excess returns documented in ESG portfolios from 2014-2020 in the U.S. and Japan.

### What is ESG Quant?
Quantitative equity investing that utilizes ESG information as risk factors or alpha signals, often implemented within systematic trading or quantitative trading approaches.

### ESG as Risk Factors
ESG information can be treated as risk factors with stable cross-sectional correlations to returns, providing:
- **Diversification benefits**: Low correlation to traditional factors (momentum, value, quality)
- **Risk mitigation**: Better downside protection
- **Long-term alpha**: Sustainability-driven returns

### Data Providers
- **LSEG (London Stock Exchange Group)**: ESG data for portfolio analysis, screening, quant models
- **Nasdaq ESG Hub**: Global ESG data platform
- **Bloomberg**: ESG scores and analytics
- **MSCI**: ESG ratings and research
- **Refinitiv**: ESG metrics
- **Sustainalytics**: ESG risk ratings

### Integration Approaches

1. **Factor Integration**
   - ESG scores as complementary factors alongside value, momentum, quality
   - Portfolio construction with ESG constraints
   - Risk model enhancement with ESG factors

2. **Screening & Tilting**
   - Negative screening (exclude low ESG scores)
   - Positive tilting (overweight high ESG scores)
   - Best-in-class selection within sectors

3. **Alpha Signals**
   - ESG momentum (improving ESG scores)
   - ESG controversies (event-driven)
   - ESG factor spreads (high vs low ESG)

### Data Quality Challenges

**Critical Issue**: ESG scores vary significantly across providers
- Each provider measures different aspects
- Different weighting methodologies
- Low correlation between provider scores
- **Solution**: Aggregate multiple providers for more robust signals

### Integration Requirements

1. **Data Schema**
   ```rust
   pub struct ESGData {
       pub timestamp: DateTime<Utc>,
       pub symbol: String,
       pub provider: String,            // "msci", "sustainalytics", "lseg"
       pub esg_score: f64,              // aggregate ESG score
       pub environmental_score: f64,
       pub social_score: f64,
       pub governance_score: f64,
       pub controversy_level: Option<String>,
       pub rating_change: Option<f64>,  // for ESG momentum
   }
   ```

2. **Backtesting Considerations**
   - **Point-in-time data**: ESG scores change as methodologies update
   - **Restatement handling**: Historical score revisions
   - **Provider selection**: Which ESG provider to use
   - **Aggregation methods**: Combining multiple ESG providers

3. **Framework Integration**
   - ESG as feature in ML models
   - ESG constraints in portfolio optimization
   - ESG-based screening filters
   - ESG factor construction

---

## 8. Order Flow / Market Microstructure Data

### Overview
Order flow reveals the raw mechanics of price formation: who's buying, who's absorbing, where liquidity is stacked or being pulled. Market microstructure models predict a positive relation between order flow and price because order flow communicates non-public information.

### Key Components

#### Level 2 (Order Book) Data
- **Bid/ask depth**: Quantity at each price level
- **Order book imbalance**: Buy vs sell pressure
- **Liquidity walls**: Large resting orders
- **Order cancellations**: Hidden liquidity removal

#### Time & Sales (Tick Data)
- **Individual trades**: Price, size, timestamp
- **Trade classification**: Buy vs sell initiated
- **Volume profile**: Volume at price levels
- **VWAP tracking**: Volume-weighted average price

#### Market Depth Metrics
- **Spread**: Bid-ask spread dynamics
- **Effective spread**: Actual execution cost
- **Market impact**: Price movement per trade size
- **Resilience**: Recovery speed after large trades

### Use Cases

#### Short-term Alpha
- **Momentum signals**: Order flow acceleration
- **Reversal signals**: Exhaustion patterns
- **Liquidity provision**: Market-making strategies
- **Execution optimization**: Minimize market impact

#### Risk Management
- **Liquidity risk**: Thin markets, wide spreads
- **Adverse selection**: Informed trader detection
- **Flash crash detection**: Liquidity evaporation
- **Circuit breaker triggers**: Exchange halts

### Integration Requirements

1. **Data Schema**
   ```rust
   pub struct OrderFlowData {
       pub timestamp: DateTime<Utc>,
       pub symbol: String,
       pub level2_snapshot: Option<OrderBook>,
       pub trades: Vec<Trade>,
       pub imbalance: f64,              // buy vs sell pressure
       pub spread_bps: f64,
       pub depth_5bps: f64,             // liquidity within 5bp of mid
   }

   pub struct OrderBook {
       pub bids: Vec<PriceLevel>,       // sorted descending
       pub asks: Vec<PriceLevel>,       // sorted ascending
   }

   pub struct PriceLevel {
       pub price: f64,
       pub size: f64,
       pub num_orders: u32,
   }

   pub struct Trade {
       pub timestamp: DateTime<Utc>,
       pub price: f64,
       pub size: f64,
       pub side: TradeSide,             // Buy, Sell, Unknown
   }
   ```

2. **Backtesting Challenges**
   - **Data volume**: Tick data is massive (GB/day)
   - **Timestamp precision**: Microsecond/nanosecond accuracy required
   - **Market impact modeling**: Realistic execution simulation
   - **Latency simulation**: Time to receive and process data

3. **Framework Integration**
   - High-frequency data loader (compressed storage)
   - Tick-level backtesting engine
   - Order book reconstruction
   - Latency/slippage modeling
   - Market impact models (already in Mantis)

---

## 9. Options Flow (Unusual Activity)

### Overview
Unusual options activity provides insight into "smart money" movements, with institutional investors accounting for 70-80% of total market trading volume. Monitoring large block trades and unusual flow patterns can signal impending price moves.

### Key Indicators

#### Volume-to-Open Interest Ratio
- **High ratio**: New contracts being opened (new positions)
- **Typical threshold**: 5x or more above average daily volume
- **Signal strength**: Institutional activity likely

#### Block Trades
- **Large size**: Institutional-sized orders (1000+ contracts)
- **Execution patterns**: Swept across multiple strikes/exchanges
- **Direction**: Bullish (calls) vs bearish (puts)

#### Unusual Flow Patterns
- **Repeated flow**: Multiple traders betting same direction
- **Short-dated OTM**: Out-of-the-money contracts expiring soon (high conviction)
- **Aggressive pricing**: Paying mid or ask (urgency)

### Data Providers & Tools
- **OptionStrat Flow**: Real-time unusual options tracking
- **InsiderFinance Flow**: Live options flow scanner
- **Unusual Whales**: Flow aggregation and analysis
- **Barchart**: Unusual options volume screener
- **TrendSpider**: Unusual options flow monitoring

### Integration Requirements

1. **Data Schema**
   ```rust
   pub struct OptionsFlow {
       pub timestamp: DateTime<Utc>,
       pub underlying_symbol: String,
       pub option_symbol: String,
       pub strike: f64,
       pub expiration: DateTime<Utc>,
       pub option_type: OptionType,     // Call, Put
       pub volume: u64,
       pub open_interest: u64,
       pub volume_oi_ratio: f64,
       pub premium: f64,
       pub spot_price: f64,
       pub sentiment: OptionsFlowSentiment,
       pub size_category: SizeCategory,  // Small, Medium, Large, Block
   }

   pub enum OptionsFlowSentiment {
       Bullish,
       Bearish,
       Neutral,
   }

   pub enum SizeCategory {
       Small,
       Medium,
       Large,
       Block,
   }
   ```

2. **Signal Generation**
   - **Aggressive call buying**: Bullish signal
   - **Protective put buying**: Risk-off signal
   - **Unusual straddle/strangle**: Volatility expectation
   - **Flow direction changes**: Sentiment shifts

3. **Backtesting Considerations**
   - **Intraday timing**: Options flow can be intraday signal
   - **False positives**: Not all unusual activity is predictive
   - **Multiple data confirmations**: Combine with other signals
   - **Execution feasibility**: Can you actually trade on this signal in time?

4. **Framework Integration**
   - Options data loader
   - Greeks calculation (delta, gamma, vega, theta)
   - Implied volatility surface
   - Options-specific backtesting logic

---

## Implementation Recommendations for Mantis

### Phase 1: Foundation (High Priority)

#### 1.1 Flexible Alternative Data Loader
Create a generic alternative data loading framework in `src/data.rs`:

```rust
pub trait AlternativeDataSource {
    type Item;
    fn load(&self, path: &Path, config: &DataConfig) -> Result<Vec<Self::Item>>;
    fn align_with_bars(&self, items: Vec<Self::Item>, bars: &[Bar]) -> Result<Vec<AlignedData>>;
}

pub enum AlternativeDataType {
    Sentiment(SentimentEvent),
    Satellite(SatelliteSignal),
    WebScraped(WebScrapedData),
    Transaction(TransactionAggregate),
    Weather(WeatherSignal),
    SupplyChain(SupplyChainEvent),
    ESG(ESGData),
    OrderFlow(OrderFlowData),
    OptionsFlow(OptionsFlow),
    Custom(serde_json::Value),      // Flexible JSON for unknown types
}

pub struct AlignedData {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub bar: Bar,
    pub alt_data: Vec<AlternativeDataType>,
}
```

**Benefits**:
- Single interface for all alternative data types
- Automatic alignment with OHLCV bars
- Point-in-time guarantees
- Extensible for future data types

#### 1.2 Update `StrategyContext`
Add alternative data access to strategy interface:

```rust
pub struct StrategyContext {
    // ... existing fields
    pub alt_data: HashMap<String, Vec<AlternativeDataType>>,
}

impl StrategyContext {
    pub fn get_sentiment(&self, symbol: &str) -> Option<&SentimentEvent> { ... }
    pub fn get_satellite_signal(&self, symbol: &str) -> Option<&SatelliteSignal> { ... }
    pub fn get_esg_data(&self, symbol: &str) -> Option<&ESGData> { ... }
    // ... etc
}
```

#### 1.3 Feature Engineering Pipeline Enhancement
Extend `src/features.rs` to handle alternative data:

```rust
pub struct AlternativeFeatures {
    pub sentiment_score: Option<f64>,
    pub sentiment_change_1d: Option<f64>,
    pub parking_lot_vehicles_yoy: Option<f64>,
    pub job_postings_change: Option<f64>,
    pub transaction_volume_growth: Option<f64>,
    pub esg_score: Option<f64>,
    pub weather_anomaly: Option<f64>,
    pub supply_chain_disruption: Option<bool>,
}

impl AlternativeFeatures {
    pub fn from_alt_data(alt_data: &[AlternativeDataType], lookback: usize) -> Self { ... }
}
```

### Phase 2: Specific Data Type Implementations (Medium Priority)

Prioritize based on user demand:

1. **Sentiment Data** (Highest demand, easiest to integrate)
   - CSV loader for pre-processed sentiment scores
   - Integration with existing feature pipeline
   - Example sentiment-based strategy

2. **ESG Data** (Growing demand, straightforward)
   - CSV loader for ESG scores from major providers
   - ESG-based portfolio screening
   - ESG momentum strategy example

3. **Weather Data** (Commodity traders)
   - Weather API integration (OpenWeather, Climavision)
   - Degree-day calculations (HDD/CDD)
   - Commodity-weather correlation analysis

4. **Web Scraping** (Most flexible)
   - Generic JSON loader for scraped data
   - Normalization utilities
   - Job postings example strategy

### Phase 3: Advanced Integration (Lower Priority)

1. **Satellite Imagery** (Resource intensive)
   - Integration with pre-processed satellite signals (not raw images)
   - Geographic aggregation utilities
   - Retail parking lot example

2. **Credit Card Data** (Requires commercial partnerships)
   - Aggregated transaction loader
   - Revenue forecasting utilities
   - Earnings surprise strategy

3. **Order Flow** (High-frequency trading focus)
   - Level 2 data loader
   - Order book reconstruction
   - HFT backtesting mode

4. **Options Flow** (Specialized use case)
   - Options chain loader
   - Greeks calculation
   - Unusual activity detection

### Phase 4: Documentation & Examples

1. **Alternative Data Guide** (`docs/alternative-data.md`)
   - Overview of each data type
   - Integration tutorial
   - Best practices for backtesting with alt data

2. **Example Strategies**
   - `examples/sentiment_trading.rs`
   - `examples/esg_momentum.rs`
   - `examples/weather_commodities.rs`
   - `examples/satellite_retail.rs`

3. **Jupyter Notebooks** (if ML integration exists)
   - Alternative data feature engineering
   - Multi-modal model training (OHLCV + alt data)
   - Walk-forward validation with alt data

---

## Key Design Principles

### 1. Point-in-Time Enforcement
**Critical**: Alternative data often has publication lag separate from event timestamp.

```rust
pub struct TimestampedData {
    pub event_time: DateTime<Utc>,      // When event occurred
    pub publication_time: DateTime<Utc>, // When data became available
    pub data: AlternativeDataType,
}
```

Backtest engine must only use data available at `publication_time`, not `event_time`.

### 2. Handling Missing Data
Alternative data is often sparse (not every bar has alt data).

**Options**:
- Forward-fill last known value
- Interpolate (with caution)
- Use `Option<T>` and handle `None` gracefully
- Maintain separate "data availability" metric

### 3. Data Quality Metrics
Track data quality for each alternative source:

```rust
pub struct DataQualityMetrics {
    pub source: String,
    pub coverage_pct: f64,           // % of bars with data
    pub lag_distribution: Vec<i64>,  // seconds from event to publication
    pub outlier_rate: f64,
    pub missing_periods: Vec<(DateTime<Utc>, DateTime<Utc>)>,
}
```

### 4. Versioning & Restatements
Alternative data providers may revise historical data.

**Best Practice**:
- Store original data with ingestion timestamp
- Track revisions separately
- Flag backtests using revised vs original data
- Document data version in backtest results

### 5. Normalization & Standardization
Different providers, different scales. Standardize before use.

```rust
pub trait Normalizable {
    fn normalize(&self) -> f64;      // Scale to [0, 1] or [-1, 1]
    fn z_score(&self) -> f64;        // Standardize (mean=0, std=1)
    fn rank_normalize(&self, universe: &[Self]) -> f64; // Percentile rank
}
```

### 6. Multi-Source Aggregation
Combine signals from multiple alternative data sources:

```rust
pub struct AggregatedSignal {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub sentiment_signal: Option<f64>,
    pub fundamental_signal: Option<f64>,
    pub alternative_signal: Option<f64>,
    pub combined_signal: f64,        // Weighted combination
    pub confidence: f64,              // Based on agreement
}
```

---

## Testing & Validation

### 1. Unit Tests
- Data loader correctness
- Timestamp alignment
- Missing data handling
- Normalization functions

### 2. Integration Tests
- End-to-end backtest with alt data
- Performance (load time, memory)
- Data quality checks
- Point-in-time verification

### 3. Validation Studies
- **Sentiment**: Earnings announcement correlation
- **Satellite**: Quarterly sales prediction accuracy
- **ESG**: Factor return analysis
- **Weather**: Commodity price correlation

### 4. Benchmark Comparisons
Compare strategies with/without alternative data:
- Information ratio improvement
- Sharpe ratio lift
- Drawdown reduction
- Alpha generation

---

## Cost-Benefit Analysis

### High-Value, Lower-Cost
1. **Sentiment data**: Many free sources (Twitter, Reddit), high alpha potential
2. **ESG data**: Some free sources (CSRHub), growing importance
3. **Weather data**: Free APIs (OpenWeather), high value for commodities

### Medium-Value, Medium-Cost
1. **Web scraping**: Setup cost high, ongoing cost low
2. **Supply chain data**: Commercial providers, moderate alpha
3. **Options flow**: Commercial feeds, specialized use case

### High-Value, High-Cost
1. **Satellite imagery**: Expensive providers ($50K-$500K/year)
2. **Credit card data**: Expensive, regulatory hurdles
3. **Order flow**: Exchange fees, infrastructure costs

### ROI Priorities for Mantis
1. **Phase 1**: Generic alt data framework (enables all future work)
2. **Phase 2**: Sentiment + ESG (high demand, moderate cost)
3. **Phase 3**: Weather + Web scraping (specialized but valuable)
4. **Phase 4**: Advanced sources as user demand dictates

---

## Regulatory & Ethical Considerations

### Data Privacy
- Credit card data: Aggregated/anonymized only
- Web scraping: Respect robots.txt, Terms of Service
- Social media: Public data only, no PII

### Insider Trading
- Alternative data must be legally obtained
- "Material non-public information" boundaries
- Expert networks: Compliance required

### Data Licensing
- Commercial providers: Redistribution restrictions
- APIs: Rate limits, terms of service
- Open data: Attribution requirements

### Best Practices
1. Maintain data provenance records
2. Document data usage in backtest metadata
3. Implement access controls for sensitive data
4. Regular compliance audits

---

## Conclusion

Alternative data integration is essential for modern quantitative trading, with the market growing at 50% annually. For the Mantis backtesting framework, a phased approach is recommended:

1. **Foundation**: Build flexible alt data loading framework
2. **Quick Wins**: Sentiment + ESG (high demand, easier integration)
3. **Specialized**: Weather, web scraping, satellite (domain-specific)
4. **Advanced**: Order flow, options flow, credit card (complex, specialized)

The key is maintaining the same rigorous standards for alternative data as for traditional OHLCV data:
- Point-in-time enforcement
- Data quality tracking
- Missing data handling
- Proper normalization

By implementing these capabilities, Mantis will be positioned to support the next generation of quantitative strategies that combine traditional technical analysis with alternative data signals.

---

## Sources

### Alternative Data & Quantitative Trading
- [Future Alpha 2026](https://www.alphaevents.com/events-futurealphaglobal)
- [Best Alternative Data Providers 2026: Full Comparison Guide](https://brightdata.com/blog/web-data/best-alternative-data-providers)
- [3 Quantitative Strategies Based on Alternative Data - TenderAlpha Blog](https://www.tenderalpha.com/blog/post/quantitative-analysis/3-quantitative-strategies-based-on-alternative-data)

### Sentiment Data Integration
- [SentimenTrader: Backtesting, Indicators & Trading Strategies](https://sentimentrader.com/)
- [GitHub - risabhmishra/algotrading-sentimentanalysis-genai](https://github.com/risabhmishra/algotrading-sentimentanalysis-genai)
- [Backtesting a Sentiment Analysis Strategy for Bitcoin](https://medium.com/hackernoon/backtesting-a-sentiment-analysis-strategy-for-bitcoin-3f79ddeb86f1)
- [arXiv: Backtesting Sentiment Signals for Trading](https://arxiv.org/abs/2507.03350)
- [GitHub - binance/ai-trading-prototype-backtester](https://github.com/binance/ai-trading-prototype-backtester)

### Satellite Imagery
- [How Satellite Imagery Is Helping Hedge Funds Outperform](https://internationalbanker.com/brokerage/how-satellite-imagery-is-helping-hedge-funds-outperform/)
- [Satellite-Powered Hedge Fund Investment Strategy | SkyFi](https://skyfi.com/en/blog/satellite-powered-hedge-fund-investment-strategy)
- [How hedge funds use satellite images to beat Wall Street—and Main Street - Haas News](https://newsroom.haas.berkeley.edu/how-hedge-funds-use-satellite-images-to-beat-wall-street-and-main-street/)
- [Satellite Data For Investors: Top Alternative Data Providers - Paragon Intel](https://paragonintel.com/satellite-data-for-investors-top-alternative-data-providers/)

### Web Scraping for Alpha
- [Web Scraping for Finance 2025: Finding Alpha in Data](https://www.promptcloud.com/blog/web-scraping-for-finance-2025/)
- [Web Scraping in Finance: 8 Use Cases for Alternative Data - BOSS Publishing](https://thebossmagazine.com/post/web-scraping-in-finance-8-use-cases-for-alternative-data/)
- [3 Quantitative Strategies Based on Alternative Data - TenderAlpha Blog](https://www.tenderalpha.com/blog/post/quantitative-analysis/3-quantitative-strategies-based-on-alternative-data)

### Credit Card Transaction Data
- [Transactions API - Bank account history & credit card data | Plaid](https://plaid.com/products/transactions/)
- [Debit & Credit Card Transaction Data: Ultimate Buyer's Guide - Facteus](https://facteus.com/buying-guides/the-ultimate-guide-to-debit-and-credit-card-transaction-data)
- [A Guide to Card Transaction Data | Enigma](https://www.enigma.com/resources/blog/a-guide-to-card-transaction-data)

### Weather Data for Trading
- [How Weather Data Moves Commodity Prices: A Trader's Guide to Weather Signals](https://blog.ehab.co/how-weather-data-moves-commodity-prices-a-traders-guide-to-weather-signals)
- [Trading on Forecasts - Weather Patterns and Market Movements](https://openweather.co.uk/blog/post/weather-patterns-and-market-movements-trading-forecasts)
- [Commodity Weather Group](https://www.commoditywx.com/)
- [Accurate Weather Forecasting for Commodity Trading | Climavision](https://climavision.com/commodity-trading/)

### Supply Chain Data
- [Shipping & Supply Chain Data For Investors: Top Alternative Data Providers - Paragon Intel](https://paragonintel.com/shipping-supply-chain-data-for-investors-top-alternative-data-providers/)
- [Supply chain data: The authoritative guide for data buyers and sellers](https://www.neudata.co/blog/supply-chain-data-the-authoritative-guide-for-data-buyers-and-sellers)

### ESG Data Integration
- [ESG Data | Company Data | Data Analytics](https://www.lseg.com/en/data-analytics/financial-data/company-data/esg-data)
- [ESG Level Factor Investing Strategy - Quantpedia](https://quantpedia.com/strategies/esg-factor-investing-strategy)
- [ESG Quant - Wikipedia](https://en.wikipedia.org/wiki/ESG_Quant)
- [Quant's Look on ESG Investing Strategies - QuantPedia](https://quantpedia.com/quants-look-on-esg-investing-strategies/)

### Order Flow & Market Microstructure
- [Market Microstructure: Order Flow and Level 2 Data Analysis](https://pocketoption.com/blog/en/knowledge-base/learning/market-microstructure/)
- [Market Microstructure & Order Flow Trading - FXAN](https://forexanalysis.com/market-microstructure-order-flow-trading/)

### Options Flow
- [Unusual Stock Options Activity - Barchart.com](https://www.barchart.com/options/unusual-activity)
- [OptionStrat Flow | Real-time Unusual Options Activity](https://optionstrat.com/flow)
- [Live Options Flow and Unusual Options Activity | InsiderFinance](https://www.insiderfinance.io/flow)
- [7-Steps Guide to Read Unusual Options Activity](https://intrinio.com/blog/how-to-read-unusual-options-activity-7-easy-steps)
- [Unusual Options Activity: A Guide to Detecting Market Anomalies](https://www.luxalgo.com/blog/unusual-options-activity-a-guide-to-detecting-market-anomalies/)
