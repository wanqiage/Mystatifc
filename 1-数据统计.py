import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, levene, mannwhitneyu, chi2_contingency
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
import seaborn as sns
import chardet

class StatisticalAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("两组数据统计比较分析")
        self.root.geometry("1000x800")
        
        self.df1 = None
        self.df2 = None
        self.results = {}
        
        self.setup_ui()
    
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
    
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 文件选择部分
        file_frame = ttk.LabelFrame(main_frame, text="文件选择", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(file_frame, text="选择第一组数据", 
                  command=self.load_file1).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(file_frame, text="选择第二组数据", 
                  command=self.load_file2).grid(row=0, column=1, padx=5, pady=5)
        
        self.file1_label = ttk.Label(file_frame, text="未选择文件")
        self.file1_label.grid(row=1, column=0, padx=5, pady=2)
        self.file2_label = ttk.Label(file_frame, text="未选择文件")
        self.file2_label.grid(row=1, column=1, padx=5, pady=2)
        
        # 数据预览
        preview_frame = ttk.LabelFrame(main_frame, text="数据预览", padding="10")
        preview_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(preview_frame, text="第一组数据列:").grid(row=0, column=0, sticky=tk.W)
        self.preview1_label = ttk.Label(preview_frame, text="暂无数据")
        self.preview1_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(preview_frame, text="第二组数据列:").grid(row=1, column=0, sticky=tk.W)
        self.preview2_label = ttk.Label(preview_frame, text="暂无数据")
        self.preview2_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # 变量选择部分
        var_frame = ttk.LabelFrame(main_frame, text="分析变量选择", padding="10")
        var_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 手动选择变量按钮
        ttk.Button(var_frame, text="手动选择数值变量", 
                  command=self.manual_select_numeric).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(var_frame, text="手动选择分类变量", 
                  command=self.manual_select_categorical).grid(row=0, column=1, padx=5, pady=5)
        
        # 显示已选变量
        ttk.Label(var_frame, text="已选数值变量:").grid(row=1, column=0, sticky=tk.W)
        self.selected_numeric_label = ttk.Label(var_frame, text="无")
        self.selected_numeric_label.grid(row=1, column=0, sticky=tk.E, padx=5)
        
        ttk.Label(var_frame, text="已选分类变量:").grid(row=1, column=1, sticky=tk.W)
        self.selected_categorical_label = ttk.Label(var_frame, text="无")
        self.selected_categorical_label.grid(row=1, column=1, sticky=tk.E, padx=5)
        
        # 存储选择的变量
        self.selected_numeric_vars = []
        self.selected_categorical_vars = []
        
        # 分析按钮
        ttk.Button(main_frame, text="执行统计分析", 
                  command=self.run_analysis).grid(row=3, column=0, columnspan=2, pady=10)
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="统计分析结果", padding="10")
        result_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.result_text = tk.Text(result_frame, height=25, width=120)
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 配置权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        var_frame.columnconfigure(0, weight=1)
        var_frame.columnconfigure(1, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
    
    def manual_select_numeric(self):
        """手动选择数值变量"""
        if self.df1 is None or self.df2 is None:
            messagebox.showerror("错误", "请先选择两个数据文件！")
            return
        
        common_columns = sorted(set(self.df1.columns) & set(self.df2.columns))
        self.manual_select_variables(common_columns, "数值变量", self.selected_numeric_vars, "selected_numeric_vars")
    
    def manual_select_categorical(self):
        """手动选择分类变量"""
        if self.df1 is None or self.df2 is None:
            messagebox.showerror("错误", "请先选择两个数据文件！")
            return
        
        common_columns = sorted(set(self.df1.columns) & set(self.df2.columns))
        self.manual_select_variables(common_columns, "分类变量", self.selected_categorical_vars, "selected_categorical_vars")
    
    def manual_select_variables(self, columns, var_type, selected_list, list_name):
        """手动选择变量的通用函数"""
        select_window = tk.Toplevel(self.root)
        select_window.title(f"手动选择{var_type}")
        select_window.geometry("500x400")
        
        ttk.Label(select_window, text=f"选择要作为{var_type}分析的列:").pack(pady=10)
        
        # 显示列信息
        info_frame = ttk.Frame(select_window)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(info_frame, text="列名").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(info_frame, text="数据类型").grid(row=0, column=1, sticky=tk.W, padx=20)
        ttk.Label(info_frame, text="唯一值数量").grid(row=0, column=2, sticky=tk.W, padx=20)
        ttk.Label(info_frame, text="示例").grid(row=0, column=3, sticky=tk.W, padx=20)
        
        # 创建带滚动条的框架
        container = ttk.Frame(select_window)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 变量选择状态
        var_states = {}
        
        # 显示每个列的信息和选择框
        for i, col in enumerate(columns):
            # 获取列信息
            dtype1 = self.df1[col].dtype
            dtype2 = self.df2[col].dtype
            unique_count1 = self.df1[col].nunique()
            unique_count2 = self.df2[col].nunique()
            sample1 = str(self.df1[col].iloc[0]) if len(self.df1) > 0 else "N/A"
            sample2 = str(self.df2[col].iloc[0]) if len(self.df2) > 0 else "N/A"
            
            row_frame = ttk.Frame(scrollable_frame)
            row_frame.grid(row=i, column=0, sticky=tk.W, pady=2)
            
            # 选择框
            var = tk.BooleanVar(value=(col in selected_list))
            var_states[col] = var
            ttk.Checkbutton(row_frame, variable=var).grid(row=0, column=0, padx=5)
            
            # 列信息
            ttk.Label(row_frame, text=col, width=20).grid(row=0, column=1, sticky=tk.W)
            ttk.Label(row_frame, text=f"{dtype1}/{dtype2}", width=15).grid(row=0, column=2, sticky=tk.W)
            ttk.Label(row_frame, text=f"{unique_count1}/{unique_count2}", width=15).grid(row=0, column=3, sticky=tk.W)
            ttk.Label(row_frame, text=f"{sample1}/{sample2}", width=20).grid(row=0, column=4, sticky=tk.W)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def confirm_selection():
            selected_vars = [col for col, var in var_states.items() if var.get()]
            selected_list.clear()
            selected_list.extend(selected_vars)
            
            # 更新显示
            if list_name == "selected_numeric_vars":
                self.selected_numeric_label.config(text=f"{len(selected_vars)}个变量")
            else:
                self.selected_categorical_label.config(text=f"{len(selected_vars)}个变量")
            
            select_window.destroy()
            messagebox.showinfo("成功", f"已选择 {len(selected_vars)} 个{var_type}")
        
        ttk.Button(select_window, text="确认选择", command=confirm_selection).pack(pady=10)
    
    def load_file1(self):
        filename = filedialog.askopenfilename(
            title="选择第一组数据CSV文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.df1 = self.load_csv_with_encoding(filename)
                self.file1_label.config(text=f"已选择: {filename.split('/')[-1]}")
                columns_str = ", ".join(self.df1.columns.tolist()[:8])
                if len(self.df1.columns) > 8:
                    columns_str += "..."
                self.preview1_label.config(text=f"{self.df1.shape[0]}行 × {self.df1.shape[1]}列: {columns_str}")
                messagebox.showinfo("成功", f"第一组数据加载成功！\n数据形状: {self.df1.shape}")
                
                # 显示列信息
                print("第一组数据列信息:")
                for col in self.df1.columns:
                    unique_vals = self.df1[col].dropna().unique()
                    print(f"  {col}: {self.df1[col].dtype}, 唯一值: {len(unique_vals)}, 示例: {list(unique_vals[:3])}")
                    
            except Exception as e:
                messagebox.showerror("错误", f"读取文件失败: {str(e)}")
    
    def load_file2(self):
        filename = filedialog.askopenfilename(
            title="选择第二组数据CSV文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.df2 = self.load_csv_with_encoding(filename)
                self.file2_label.config(text=f"已选择: {filename.split('/')[-1]}")
                columns_str = ", ".join(self.df2.columns.tolist()[:8])
                if len(self.df2.columns) > 8:
                    columns_str += "..."
                self.preview2_label.config(text=f"{self.df2.shape[0]}行 × {self.df2.shape[1]}列: {columns_str}")
                messagebox.showinfo("成功", f"第二组数据加载成功！\n数据形状: {self.df2.shape}")
                
                # 显示列信息
                print("第二组数据列信息:")
                for col in self.df2.columns:
                    unique_vals = self.df2[col].dropna().unique()
                    print(f"  {col}: {self.df2[col].dtype}, 唯一值: {len(unique_vals)}, 示例: {list(unique_vals[:3])}")
                    
            except Exception as e:
                messagebox.showerror("错误", f"读取文件失败: {str(e)}")
    
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
            'mean1': np.mean(data1_clean),
            'mean2': np.mean(data2_clean),
            'std1': np.std(data1_clean, ddof=1),
            'std2': np.std(data2_clean, ddof=1),
            'normality1': norm1,
            'normality2': norm2,
            'p_norm1': p_norm1,
            'p_norm2': p_norm2,
            'variance_homogeneity': var_equal,
            'p_var': p_var
        }
        
        # 选择适当的检验方法
        if norm1 and norm2 and var_equal:
            # 参数检验：独立样本t检验
            t_stat, p_value = stats.ttest_ind(data1_clean, data2_clean)
            test_type = "t检验"
            result.update({'test_type': test_type, 'statistic': t_stat, 'p_value': p_value})
            
        elif norm1 and norm2 and not var_equal:
            # 参数检验：Welch's t检验
            t_stat, p_value = stats.ttest_ind(data1_clean, data2_clean, equal_var=False)
            test_type = "Welch's t检验"
            result.update({'test_type': test_type, 'statistic': t_stat, 'p_value': p_value})
            
        else:
            # 非参数检验：Mann-Whitney U检验
            u_stat, p_value = mannwhitneyu(data1_clean, data2_clean, alternative='two-sided')
            test_type = "Mann-Whitney U检验"
            result.update({'test_type': test_type, 'statistic': u_stat, 'p_value': p_value})
        
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
                'categories': all_categories,
                'frequencies1': freq1.to_dict(),
                'frequencies2': freq2.to_dict(),
                'percentages1': (freq1 / len(data1_clean) * 100).to_dict(),
                'percentages2': (freq2 / len(data2_clean) * 100).to_dict(),
                'test_type': '卡方检验',
                'statistic': chi2,
                'p_value': p_value
            }
            
            return "成功", result
            
        except Exception as e:
            return f"卡方检验失败: {str(e)}", None
    
    def run_analysis(self):
        if self.df1 is None or self.df2 is None:
            messagebox.showerror("错误", "请先选择两个数据文件！")
            return
        
        if not self.selected_numeric_vars and not self.selected_categorical_vars:
            messagebox.showerror("错误", "请先手动选择要分析的变量！")
            return
        
        self.results = {}
        output_text = "统计分析结果\n" + "="*80 + "\n\n"
        
        # 分析数值变量
        for var in self.selected_numeric_vars:
            status, result = self.perform_numerical_test(self.df1[var], self.df2[var], var)
            
            if status == "成功":
                self.results[var] = result
                output_text += self.format_numerical_result(result)
            else:
                output_text += f"数值变量 {var}: {status}\n\n"
        
        # 分析分类变量
        for var in self.selected_categorical_vars:
            status, result = self.perform_categorical_test(self.df1[var], self.df2[var], var)
            
            if status == "成功":
                self.results[var] = result
                output_text += self.format_categorical_result(result)
            else:
                output_text += f"分类变量 {var}: {status}\n\n"
        
        # 生成三线表
        output_text += self.generate_three_line_table()
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, output_text)
    
    def format_numerical_result(self, result):
        """格式化数值变量结果"""
        text = f"变量: {result['variable']} (数值变量)\n"
        text += f"第一组: n={result['n1']}, 均值={result['mean1']:.3f}, 标准差={result['std1']:.3f}\n"
        text += f"第二组: n={result['n2']}, 均值={result['mean2']:.3f}, 标准差={result['std2']:.3f}\n"
        text += f"正态性检验(组1): p={result['p_norm1']:.4f} {'(正态)' if result['normality1'] else '(非正态)'}\n"
        text += f"正态性检验(组2): p={result['p_norm2']:.4f} {'(正态)' if result['normality2'] else '(非正态)'}\n"
        text += f"方差齐性检验: p={result['p_var']:.4f} {'(齐性)' if result['variance_homogeneity'] else '(非齐性)'}\n"
        text += f"检验方法: {result['test_type']}\n"
        text += f"检验统计量: {result['statistic']:.4f}\n"
        text += f"P值: {result['p_value']:.4f}\n"
        significance = "显著" if result['p_value'] < 0.05 else "不显著"
        text += f"统计显著性: {significance} (α=0.05)\n"
        text += "-"*60 + "\n\n"
        return text
    
    def format_categorical_result(self, result):
        """格式化分类变量结果"""
        text = f"变量: {result['variable']} (分类变量)\n"
        text += f"第一组: n={result['n1']}\n"
        text += f"第二组: n={result['n2']}\n"
        
        for category in result['categories']:
            freq1 = result['frequencies1'].get(category, 0)
            freq2 = result['frequencies2'].get(category, 0)
            perc1 = result['percentages1'].get(category, 0)
            perc2 = result['percentages2'].get(category, 0)
            text += f"  类别 {category}: 组1={freq1} ({perc1:.1f}%), 组2={freq2} ({perc2:.1f}%)\n"
        
        text += f"检验方法: {result['test_type']}\n"
        text += f"卡方值: {result['statistic']:.4f}\n"
        text += f"P值: {result['p_value']:.4f}\n"
        significance = "显著" if result['p_value'] < 0.05 else "不显著"
        text += f"统计显著性: {significance} (α=0.05)\n"
        text += "-"*60 + "\n\n"
        return text
    
    def generate_three_line_table(self):
        """生成统计三线表（按照指定格式）"""
        if not self.results:
            return "\n无法生成三线表：无有效结果"
        
        table = "\n统计三线表\n" + "="*80 + "\n"
        table += f"{'variables':<15}{'level':<10}{'0':<20}{'1':<20}{'p':<10}\n"
        table += "-"*80 + "\n"
        
        # 添加样本量行
        n1 = next(iter(self.results.values()))['n1'] if self.results else 0
        n2 = next(iter(self.results.values()))['n2'] if self.results else 0
        table += f"{'n':<15}{'':<10}{n1:<20}{n2:<20}{'':<10}\n"
        
        # 添加其他变量
        for var_name, result in self.results.items():
            if result['type'] == 'numerical':
                # 数值变量：显示均值±标准差
                mean_std1 = f"{result['mean1']:.2f} ± {result['std1']:.2f}"
                mean_std2 = f"{result['mean2']:.2f} ± {result['std2']:.2f}"
                p_value = f"{result['p_value']:.3f}" if not np.isnan(result['p_value']) else "NaN"
                table += f"{var_name:<15}{'':<10}{mean_std1:<20}{mean_std2:<20}{p_value:<10}\n"
            
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
                        table += f"{var_name:<15}{level_str:<10}{group1_str:<20}{group2_str:<20}{p_value:<10}\n"
                    else:
                        # 后续行只显示类别
                        table += f"{'':<15}{level_str:<10}{group1_str:<20}{group2_str:<20}{'':<10}\n"
        
        table += "-"*80 + "\n"
        table += "注: 数值变量显示为均值±标准差，分类变量显示为频数(百分比)\n"
        
        # 添加统计检验方法说明
        test_methods = set()
        for result in self.results.values():
            test_methods.add(result['test_type'])
        
        if test_methods:
            table += f"检验方法: {', '.join(test_methods)}\n"
        
        return table
    
    def run(self):
        self.root.mainloop()

# 运行应用程序
if __name__ == "__main__":
    # 如果chardet不可用，使用简化版本
    try:
        import chardet
    except ImportError:
        print("警告: chardet模块未安装，将使用基本编码检测")
        def simple_encoding_detector(file_path):
            return 'gbk'  # 中文环境常用编码
        
        StatisticalAnalyzer.detect_encoding = simple_encoding_detector
    
    app = StatisticalAnalyzer()
    app.run()