import json
with open('c:\\Users\\saima\\OneDrive\\Documents\\My_MSA_Project\\MSA_Research_Colab.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('cggm_hybrid_best.pt', 'robust_hybrid_best.pt')
with open('c:\\Users\\saima\\OneDrive\\Documents\\My_MSA_Project\\MSA_Research_Colab.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)
