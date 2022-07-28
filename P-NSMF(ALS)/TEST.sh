# ML1M
python main.py -dataset="ML1M-1" -omega=4 -group_size=20 -lambda_=0.01 -T=500 -d=20 -topK=5 -n=6040 -m=3952 -train_path="ML1M-TXT-FORMAT/ML1M-copy1-train" -test_path="ML1M-TXT-FORMAT/ML1M-copy1-test"
python main.py -dataset="ML1M-2" -omega=4 -group_size=20 -lambda_=0.01 -T=500 -d=20 -topK=5 -n=6040 -m=3952 -train_path="ML1M-TXT-FORMAT/ML1M-copy2-train" -test_path="ML1M-TXT-FORMAT/ML1M-copy2-test"
python main.py -dataset="ML1M-3" -omega=4 -group_size=20 -lambda_=0.01 -T=500 -d=20 -topK=5 -n=6040 -m=3952 -train_path="ML1M-TXT-FORMAT/ML1M-copy3-train" -test_path="ML1M-TXT-FORMAT/ML1M-copy3-test"
# Netflix5K5K
python main.py -dataset="NF5k5k-1" -omega=4 -group_size=20 -lambda_=0.01 -T=500 -d=20 -topK=5 -n=5000 -m=5000 -train_path="Netflix5K5K-TXT-FORAMT/NF5kUsers5kItemsHalfHalf-copy1-train" -test_path="Netflix5K5K-TXT-FORAMT/NF5kUsers5kItemsHalfHalf-copy1-test"
python main.py -dataset="NF5k5k-2" -omega=4 -group_size=20 -lambda_=0.01 -T=500 -d=20 -topK=5 -n=5000 -m=5000 -train_path="Netflix5K5K-TXT-FORAMT/NF5kUsers5kItemsHalfHalf-copy2-train" -test_path="Netflix5K5K-TXT-FORAMT/NF5kUsers5kItemsHalfHalf-copy2-test"
python main.py -dataset="NF5k5k-3" -omega=4 -group_size=20 -lambda_=0.01 -T=500 -d=20 -topK=5 -n=5000 -m=5000 -train_path="Netflix5K5K-TXT-FORAMT/NF5kUsers5kItemsHalfHalf-copy3-train" -test_path="Netflix5K5K-TXT-FORAMT/NF5kUsers5kItemsHalfHalf-copy3-test"
# XING5K5K
python main.py -dataset="XING5K5K-1" -omega=6 -group_size=20 -lambda_=0 -T=500 -d=20 -topK=5 -n=5000 -m=5000 -train_path="XING5K5K/copy1.train" -test_path="XING5K5K/copy1.test"
python main.py -dataset="XING5K5K-2" -omega=6 -group_size=20 -lambda_=0 -T=500 -d=20 -topK=5 -n=5000 -m=5000 -train_path="XING5K5K/copy2.train" -test_path="XING5K5K/copy2.test"
python main.py -dataset="XING5K5K-3" -omega=6 -group_size=20 -lambda_=0 -T=500 -d=20 -topK=5 -n=5000 -m=5000 -train_path="XING5K5K/copy3.train" -test_path="XING5K5K/copy3.test"
# Amazon_Kindle_Store
python main.py -dataset="AmazonKS-1" -omega=6 -group_size=20 -lambda_=0 -T=500 -d=20 -topK=5 -n=9862 -m=11298 -train_path="Amazon_Kindle_Store-TXT-FORMAT/Amazon_Kindle_Store-copy1-train" -test_path="Amazon_Kindle_Store-TXT-FORMAT/Amazon_Kindle_Store-copy1-test"
python main.py -dataset="AmazonKS-2" -omega=6 -group_size=20 -lambda_=0 -T=500 -d=20 -topK=5 -n=9862 -m=11298 -train_path="Amazon_Kindle_Store-TXT-FORMAT/Amazon_Kindle_Store-copy2-train" -test_path="Amazon_Kindle_Store-TXT-FORMAT/Amazon_Kindle_Store-copy2-test"
python main.py -dataset="AmazonKS-3" -omega=6 -group_size=20 -lambda_=0 -T=500 -d=20 -topK=5 -n=9862 -m=11298 -train_path="Amazon_Kindle_Store-TXT-FORMAT/Amazon_Kindle_Store-copy3-train" -test_path="Amazon_Kindle_Store-TXT-FORMAT/Amazon_Kindle_Store-copy3-test"
