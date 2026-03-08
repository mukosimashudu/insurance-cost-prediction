import plotly.express as px
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [10, 15, 13, 17, 20]
})

# Create a simple plot
fig = px.line(df, x='x', y='y', title='Test Plot')
print("✅ Plotly installed successfully!")
print(f"Plotly version: {px.__version__}")