from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, levene, mannwhitneyu, chi2_contingency
import os
import chardet
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局变量存储数据
df1 = None
df2 = None
selected_numeric_vars = []
selected_categorical_vars = []

class StatisticalAnalyzer:
    def __init__(self):
        pass

    def detect_encoding(self, file_path):
        """检测文件编码"""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            print(f"检测到编码: {encoding}, 置信度: {confidence}")
            return encoding

    def load_csv_with_encoding(self, file_path):
        """尝试多种编码方式读取CSV文件"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'iso-8859-1']

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"成功使用 {encoding} 编码读取文件")
                return df
            except UnicodeDecodeError:
                print(f"{encoding} 编码失败，尝试下一种...")
                continue
            except Exception as e:
                print(f"{encoding} 编码读取错误: {e}")
                continue

        # 如果常见编码都失败，尝试自动检测
        try:
            detected_encoding = self.detect_encoding(file_path)
            if detected_encoding:
                df = pd.read_csv(file_path, encoding=detected_encoding)
                print(f"使用检测到的编码 {detected_encoding} 成功读取文件")
                return df
        except Exception as e:
            print(f"自动检测编码也失败: {e}")

        raise ValueError("无法读取文件，请检查文件格式或编码")

    def check_normality(self, data):
        """检查数据正态性"""
        if len(data) < 3:
            return False, 1.0
        try:
            stat, p_value = shapiro(data)
            return p_value > 0.05, p_value
        except:
            return False, 1.0

    def check_variance_homogeneity(self, data1, data2):
        """检查方差齐性"""
        if len(data1) < 2 or len(data2) < 2:
            return False, 1.0
        try:
            stat, p_value = levene(data1, data2)
            return p_value > 0.05, p_value
        except:
            return False, 1.0

    def perform_numerical_test(self, data1, data2, var_name):
        """执行数值变量的统计检验"""
        # 移除缺失值
        data1_clean = data1.dropna()
        data2_clean = data2.dropna()

        if len(data1_clean) < 2 or len(data2_clean) < 2:
            return "样本量不足", None

        # 检查正态性
        norm1, p_norm1 = self.check_normality(data1_clean)
        norm2, p_norm2 = self.check_normality(data2_clean)

        # 检查方差齐性
        var_equal, p_var = self.check_variance_homogeneity(data1_clean, data2_clean)

        result = {
            'variable': var_name,
            'type': 'numerical',
            'n1': len(data1_clean),
            'n2': len(data2_clean),
            'mean1': float(np.mean(data1_clean)),
            'mean2': float(np.mean(data2_clean)),
            'std1': float(np.std(data1_clean, ddof=1)),
            'std2': float(np.std(data2_clean, ddof=1)),
            'normality1': norm1,
            'normality2': norm2,
            'p_norm1': float(p_norm1),
            'p_norm2': float(p_norm2),
            'variance_homogeneity': var_equal,
            'p_var': float(p_var)
        }

        # 选择适当的检验方法
        if norm1 and norm2 and var_equal:
            # 参数检验：独立样本t检验
            t_stat, p_value = stats.ttest_ind(data1_clean, data2_clean)
            test_type = "t检验"
            result.update({'test_type': test_type, 'statistic': float(t_stat), 'p_value': float(p_value)})

        elif norm1 and norm2 and not var_equal:
            # 参数检验：Welch's t检验
            t_stat, p_value = stats.ttest_ind(data1_clean, data2_clean, equal_var=False)
            test_type = "Welch's t检验"
            result.update({'test_type': test_type, 'statistic': float(t_stat), 'p_value': float(p_value)})

        else:
            # 非参数检验：Mann-Whitney U检验
            u_stat, p_value = mannwhitneyu(data1_clean, data2_clean, alternative='two-sided')
            test_type = "Mann-Whitney U检验"
            result.update({'test_type': test_type, 'statistic': float(u_stat), 'p_value': float(p_value)})

        return "成功", result

    def perform_categorical_test(self, data1, data2, var_name):
        """执行分类变量的卡方检验"""
        # 移除缺失值
        data1_clean = data1.dropna()
        data2_clean = data2.dropna()

        if len(data1_clean) < 2 or len(data2_clean) < 2:
            return "样本量不足", None

        # 创建列联表
        try:
            # 转换为字符串类型以确保一致性
            data1_str = data1_clean.astype(str).str.strip()
            data2_str = data2_clean.astype(str).str.strip()

            # 获取所有类别
            all_categories = sorted(set(data1_str.unique()) | set(data2_str.unique()))

            # 计算频数
            freq1 = data1_str.value_counts().reindex(all_categories, fill_value=0)
            freq2 = data2_str.value_counts().reindex(all_categories, fill_value=0)

            # 创建列联表
            contingency_table = pd.DataFrame({
                'Group1': freq1,
                'Group2': freq2
            }).T

            # 检查列联表是否有效
            if contingency_table.sum().sum() == 0:
                return "列联表为空", None

            # 执行卡方检验
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            result = {
                'variable': var_name,
                'type': 'categorical',
                'n1': len(data1_clean),
                'n2': len(data2_clean),
                'categories': list(all_categories),
                'frequencies1': freq1.to_dict(),
                'frequencies2': freq2.to_dict(),
                'percentages1': (freq1 / len(data1_clean) * 100).to_dict(),
                'percentages2': (freq2 / len(data2_clean) * 100).to_dict(),
                'test_type': '卡方检验',
                'statistic': float(chi2),
                'p_value': float(p_value)
            }

            return "成功", result

        except Exception as e:
            return f"卡方检验失败: {str(e)}", None

    def generate_three_line_table(self, results):
        """生成统计三线表"""
        if not results:
            return {"error": "无有效结果"}

        table_data = []

        # 添加样本量行
        if results:
            first_result = next(iter(results.values()))
            table_data.append({
                'variables': 'n',
                'level': '',
                'group0': first_result['n1'],
                'group1': first_result['n2'],
                'p': ''
            })

        # 添加其他变量
        for var_name, result in results.items():
            if result['type'] == 'numerical':
                # 数值变量：显示均值±标准差
                mean_std1 = f"{result['mean1']:.2f} ± {result['std1']:.2f}"
                mean_std2 = f"{result['mean2']:.2f} ± {result['std2']:.2f}"
                p_value = f"{result['p_value']:.3f}" if not np.isnan(result['p_value']) else "NaN"
                table_data.append({
                    'variables': var_name,
                    'level': '',
                    'group0': mean_std1,
                    'group1': mean_std2,
                    'p': p_value
                })
            else:
                # 分类变量：显示每个类别的频数（百分比）
                categories = result['categories']
                for i, category in enumerate(categories):
                    freq1 = result['frequencies1'].get(category, 0)
                    freq2 = result['frequencies2'].get(category, 0)
                    perc1 = result['percentages1'].get(category, 0)
                    perc2 = result['percentages2'].get(category, 0)

                    level_str = str(category)
                    group1_str = f"{freq1} ({perc1:.1f})"
                    group2_str = f"{freq2} ({perc2:.1f})"

                    if i == 0:
                        # 第一行显示变量名和p值
                        p_value = f"{result['p_value']:.3f}" if not np.isnan(result['p_value']) else "NaN"
                        table_data.append({
                            'variables': var_name,
                            'level': level_str,
                            'group0': group1_str,
                            'group1': group2_str,
                            'p': p_value
                        })
                    else:
                        # 后续行只显示类别
                        table_data.append({
                            'variables': '',
                            'level': level_str,
                            'group0': group1_str,
                            'group1': group2_str,
                            'p': ''
                        })

        return {"data": table_data}

analyzer = StatisticalAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    global df1, df2

    try:
        file = request.files['file']
        group = request.form['group']

        if file.filename == '':
            return jsonify({'error': '未选择文件'})

        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 读取CSV文件
            df = analyzer.load_csv_with_encoding(filepath)

            # 获取列信息
            columns_info = []
            for col in df.columns:
                unique_vals = df[col].dropna().unique()
                columns_info.append({
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'unique_count': len(unique_vals),
                    'sample': str(unique_vals[0]) if len(unique_vals) > 0 else 'N/A'
                })

            data_info = {
                'shape': df.shape,
                'columns': columns_info
            }

            if group == '1':
                df1 = df
            else:
                df2 = df

            return jsonify({'success': True, 'data': data_info})
        else:
            return jsonify({'error': '请上传CSV文件'})

    except Exception as e:
        return jsonify({'error': f'文件处理失败: {str(e)}'})

@app.route('/get_common_columns')
def get_common_columns():
    global df1, df2

    if df1 is None or df2 is None:
        return jsonify({'error': '请先上传两个数据文件'})

    common_columns = sorted(set(df1.columns) & set(df2.columns))

    columns_info = []
    for col in common_columns:
        dtype1 = df1[col].dtype
        dtype2 = df2[col].dtype
        unique_count1 = df1[col].nunique()
        unique_count2 = df2[col].nunique()
        sample1 = str(df1[col].iloc[0]) if len(df1) > 0 else "N/A"
        sample2 = str(df2[col].iloc[0]) if len(df2) > 0 else "N/A"

        columns_info.append({
            'name': col,
            'dtype1': str(dtype1),
            'dtype2': str(dtype2),
            'unique_count1': int(unique_count1),
            'unique_count2': int(unique_count2),
            'sample1': sample1,
            'sample2': sample2
        })

    return jsonify({'columns': columns_info})

@app.route('/select_variables', methods=['POST'])
def select_variables():
    global selected_numeric_vars, selected_categorical_vars

    data = request.get_json()
    selected_numeric_vars = data.get('numeric', [])
    selected_categorical_vars = data.get('categorical', [])

    return jsonify({
        'success': True,
        'numeric_count': len(selected_numeric_vars),
        'categorical_count': len(selected_categorical_vars)
    })

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    global df1, df2, selected_numeric_vars, selected_categorical_vars

    if df1 is None or df2 is None:
        return jsonify({'error': '请先选择两个数据文件！'})

    if not selected_numeric_vars and not selected_categorical_vars:
        return jsonify({'error': '请先选择要分析的变量！'})

    results = {}

    # 分析数值变量
    for var in selected_numeric_vars:
        if var in df1.columns and var in df2.columns:
            status, result = analyzer.perform_numerical_test(df1[var], df2[var], var)
            if status == "成功":
                results[var] = result

    # 分析分类变量
    for var in selected_categorical_vars:
        if var in df1.columns and var in df2.columns:
            status, result = analyzer.perform_categorical_test(df1[var], df2[var], var)
            if status == "成功":
                results[var] = result

    # 生成三线表
    table_result = analyzer.generate_three_line_table(results)

    return jsonify({
        'success': True,
        'results': results,
        'table': table_result
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)