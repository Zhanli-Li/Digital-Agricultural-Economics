# Digital-Agricultural-Economics / 数字农业经济
**Author / 作者** ：[Zhanli Li / 李展利](https://zhanli-li.github.io)

If you find any problem, feel free open a issue in github, I will do my best to solve it.
## English Version

This repository contains the final project for the "Principles and Applications of Digital Economy (Bilingual Course)" at Wenlan School of Business, Zhongnan University of Economics and Law.

The project focuses on applying digital economy principles and technology to agricultural economics, specifically exploring time-series forecasting of agricultural product sales using various deep learning models.

### Project Structure

*   **/code**: Contains the Python scripts and Jupyter notebooks for data processing, model implementation (LSTM, KAN-LSTM, xLSTM), and result generation.
    *   `KAN-LSTM.py`: Implementation of the KAN-LSTM model.
    *   `LSTM.py`: Implementation of the standard LSTM model.
    *   `xLSTM.py`: Implementation of an experimental xLSTM model.
    *   `optimization.ipynb`: Jupyter notebook likely used for profit optimization and experimentation.
*   **/data**: Contains the dataset(s) used for training and testing the models (e.g., `sales_threeyear.xlsx`).
*   **/paper**:Contains the final report or paper related to this project.
*   `LICENSE`: The Apache 2.0 License file for this project.

### Models Explored

The project implements and compares the following models for sales forecasting:
*   Long Short-Term Memory (LSTM)
*   KAN-LSTM (LSTM with Kolmogorov-Arnold Networks concepts)
*   xLSTM (an experimental variant of LSTM)

### Core Technologies
* Python
* Pandas
* NumPy
* PyTorch
* Scikit-learn
* Matplotlib

---

## 中文版本

本仓库包含中南财经政法大学文澜学院《数字经济原理与应用（双语）》的课程结课项目。

该项目致力于将数字经济原理和技术应用于农业经济学领域，特别是利用多种深度学习模型对农产品销量进行时间序列预测。

### 项目结构

*   **/code**: 包含数据处理、模型实现（LSTM, KAN-LSTM, xLSTM）及结果生成的 Python 脚本和 Jupyter notebook。
    *   `KAN-LSTM.py`: KAN-LSTM 模型的实现。
    *   `LSTM.py`: 标准 LSTM 模型的实现。
    *   `xLSTM.py`: xLSTM 模型的实现。
    *   `optimization.ipynb`: 用于利润优化实验的notebook。
*   **/data**: 包含用于训练和测试模型的数据集 (例如 `sales_threeyear.xlsx`)。
*   **/paper**: 包含与此项目相关的最终报告或论文。
*   `LICENSE`: 本项目的 Apache 2.0 许可证文件。

### 模型探索

项目实现并比较了以下模型用于销量预测：
*   长短期记忆网络 (LSTM)
*   KAN-LSTM (结合了 Kolmogorov-Arnold 网络思想的 LSTM)
*   xLSTM (一种实验性的 LSTM 变体)

### 依赖库
* Pandas
* NumPy
* PyTorch
* Scikit-learn
* Matplotlib

---
