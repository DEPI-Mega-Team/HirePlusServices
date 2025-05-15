import streamlit as st
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
df = pd.read_excel(os.path.join(BASE_DIR, "Cleaned_Wuzzuf.xlsx"))

# Set the title of the Streamlit app
st.title("Hire+ Job Market Insights and Trends")

# Color palette
custom_color_palette = [
    "#12181E", "#171F27", "#1E2832", "#263441", "#2A3947", "#55616C",
    "#707A84", "#798994", "#8596A3", "#9DAABB", "#BDC2C6", "#EAEBED"
]

# Top Job Titles
st.subheader("Top 10 Job Titles by Demand")
top_job_titles = df['job_title'].value_counts().nlargest(10).reset_index()
top_job_titles.columns = ['job_title', 'count']
fig1 = px.bar(top_job_titles, x='count', y='job_title', orientation='h',
              color='job_title', color_discrete_sequence=custom_color_palette,
              title='Top 10 Job Titles by Demand')
st.plotly_chart(fig1)

# Job Distribution Map
st.subheader("Job Distribution by City in Egypt")
city_distribution = df['city'].value_counts().reset_index()
city_distribution.columns = ['city', 'count']

egypt_city_coords = pd.DataFrame({
    'city': ["Cairo", "Giza", "Alexandria", "Port Said", "Suez", "Damietta", "Dakahlia", "Sharkia", "Qalyubia", "Kafr El Sheikh",
             "Beheira", "Ismailia", "Gharbia", "Monufia", "Faiyum", "Beni Suef", "Minya", "Assiut", "Sohag", "Qena", "Luxor",
             "Aswan", "Red Sea", "New Valley", "Matrouh", "North Sinai", "South Sinai"],
    'lat': [30.0444, 30.0131, 31.2001, 31.2653, 29.9668, 31.4165, 31.0364, 30.7320, 30.3339, 31.3085,
            30.8480, 30.6043, 30.8754, 30.5972, 29.3084, 29.0661, 28.1099, 27.1800, 26.5560, 26.1551,
            25.6872, 24.0889, 27.2579, 25.3250, 31.3525, 30.3060, 28.1099],
    'lon': [31.2357, 31.2089, 29.9187, 32.3019, 32.5498, 31.8100, 31.3807, 31.7195, 31.2421, 30.9336,
            30.3400, 32.2723, 30.8195, 30.9876, 30.8467, 31.0978, 30.7503, 31.1837, 31.6948, 32.7160,
            32.6396, 32.8998, 33.8129, 30.5467, 27.2289, 33.7992, 33.3000]
})

city_distribution = city_distribution.merge(egypt_city_coords, on='city', how='left')
fig2 = px.scatter_mapbox(city_distribution, lat="lat", lon="lon", size="count", color="count",
                         hover_name="city", size_max=40, zoom=5, mapbox_style="carto-positron",
                         title="Job Distribution by City in Egypt")
st.plotly_chart(fig2)

# Average Salary
st.subheader("Average Annual Salary by Job Title")
df['annual_salary'] = df['annual_salary'].fillna(0)
top_titles_salary = df['job_title'].value_counts().nlargest(10).index
avg_salary = df[df['job_title'].isin(top_titles_salary)].groupby('job_title')['annual_salary'].mean().sort_values(ascending=False).reset_index()
fig3 = px.bar(avg_salary, x='annual_salary', y='job_title', orientation='h',
              color='job_title', color_discrete_sequence=custom_color_palette,
              title='Average Annual Salary by Job Title')
st.plotly_chart(fig3)

# Job Post Trends
st.subheader("Job Post Trends Over Time")
df['post_date'] = pd.to_datetime(df['post_date'])
df['month'] = df['post_date'].dt.to_period('M').astype(str)
monthly_trend = df.groupby('month').size().reset_index(name='job_posts')
fig4 = px.line(monthly_trend, x='month', y='job_posts', markers=True,
               color_discrete_sequence=[custom_color_palette[4]],
               title='Job Post Trends Over Time')
st.plotly_chart(fig4)

# Vacancies by Category
st.subheader("Number of Vacancies by Job Category")
vacancies_by_category = df.groupby('Category')['num_vacancies'].sum().sort_values(ascending=False).reset_index()
fig5 = px.bar(vacancies_by_category, x='Category', y='num_vacancies',
              color='Category', color_discrete_sequence=custom_color_palette,
              title='Number of Vacancies by Job Category')
st.plotly_chart(fig5)

# Metrics
st.subheader("Job Market Metrics")
total_postings = len(df)
total_vacancies = df['num_vacancies'].sum()

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Total Job Postings", value=total_postings)
with col2:
    st.metric(label="Total Vacancies", value=total_vacancies)
#