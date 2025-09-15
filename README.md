# USA Medical Industry Analytics Dashboard

A comprehensive Python-based dashboard for analyzing the US healthcare industry, including spending trends, workforce data, facility information, and company analytics using real public data sources.

## Features

- **Healthcare Spending Analysis**: Track total spending, trends over time, and category breakdowns
- **Workforce Analytics**: Healthcare employment by state and workforce statistics
- **Facility Data**: Hospital beds per capita and healthcare facility information
- **Company Insights**: Top healthcare companies by revenue
- **Interactive Visualizations**: Built with Plotly and Dash for dynamic charts and maps
- **Real-time Data**: Integration with public APIs (CMS, BLS, CDC, Census)

## Data Sources

- **CMS (Centers for Medicare & Medicaid Services)**: Healthcare spending and Medicare data
- **BLS (Bureau of Labor Statistics)**: Employment and wage data
- **CDC (Centers for Disease Control)**: Health statistics and facility data
- **Census Bureau**: Population and demographic data

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd medical_usa_base
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:8050`

## Project Structure

```
medical_usa_base/
├── app.py              # Main Dash application
├── data_fetcher.py     # Data collection and processing
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Key Metrics Displayed

- **Total Healthcare Spending**: $4.3 trillion annually
- **Healthcare Facilities**: 6,090+ hospitals nationwide
- **Healthcare Workers**: 22+ million employed
- **Cost per Capita**: $12,914 average spending per person

## Charts and Visualizations

1. **Healthcare Spending Trends**: Line chart showing spending growth over time
2. **Spending by Category**: Pie chart breaking down healthcare expenditures
3. **Employment Map**: Choropleth map of healthcare jobs by state
4. **Top Companies**: Horizontal bar chart of largest healthcare companies
5. **Hospital Beds**: Bar chart of beds per 1000 population by state

## Technologies Used

- **Python 3.8+**
- **Dash & Plotly**: Interactive web applications and visualizations
- **Pandas & NumPy**: Data processing and analysis
- **Requests**: API data fetching
- **Bootstrap**: Responsive UI components

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.