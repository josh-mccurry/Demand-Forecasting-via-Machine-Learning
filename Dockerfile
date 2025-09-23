FROM jupyter/scipy-notebook:latest
WORKDIR /home/jovyan/.local/
RUN pip install jupyter
EXPOSE 8888
ENV NAME Demand
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY Capstone_Demand_Forecasting.ipynb .
COPY __init__.py .
COPY data_handler.py .
COPY modeler.py .
COPY predictor.py .
COPY visualizer.py .
COPY demand_data.csv .
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]


