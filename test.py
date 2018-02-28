# _*_ coding:utf-8 _*_
from get_papers import get_papers
import pickle
# import visualize_papers


def list_to_md(data_list):
    md_text = '|id|titile|\n|:---:|:---|'
    for dl in data_list:
        new_low = '\n|' + dl[0] + '|[' + dl[1] + '](' + dl[3] + ')|'
        md_text += new_low
    return md_text


def main():
    papers = get_papers()
    print(papers)
    # TODO:these_pickles are for debug.
    with open('iclr_2018_papers.pickle', mode='wb') as f:
        pickle.dump(papers, f)
    with open('iclr_2018_papers.pickle', mode='rb') as f:
        papers = pickle.load(f)
    text = list_to_md(papers)
    with open('papers.md', 'w') as f:
        f.write(text)  # 引数の文字列をファイルに書き込む
    # visualize_papers(papers)
# papers =[[id,title,abstruct,html],]


if __name__ == '__main__':
    main()
