# Beyond Deep Reinforcement Learning: A Tutorial on Generative Diffusion Models in Network Optimization

Generative Diffusion Models (GDMs) have emerged as a transformative force in the realm of Generative Artificial Intelligence (GAI), demonstrating their versatility and efficacy across a variety of applications. The ability to model complex data distributions and generate high-quality samples has made GDMs particularly effective in tasks such as image generation and reinforcement learning. Furthermore, their iterative nature, which involves a series of noise addition and denoising steps, is a powerful and unique approach to learning and generating data.

This repository contains the code accompanying the paper 

> **"Beyond Deep Reinforcement Learning: A Tutorial on Generative Diffusion Models in Network Optimization"**

Authored by *Hongyang Du, Ruichen Zhang, Yinqiu Liu, Jiacheng Wang, Yijing Lin, Zonghang Li, Dusit Niyato, Jiawen Kang, Zehui Xiong, Shuguang Cui, Bo Ai, Haibo Zhou, and Dong In Kim*, submitted to *IEEE Communications Surveys & Tutorials*.

The paper can be found at [ArXiv](https://arxiv.org/abs/2308.05384).

## ⚡ Structure of Our Tutorial
<img src="images/0.png" width = "100%">

We initiate our discussion with the foundational knowledge of GDM and the motivation behind their applications in network optimization. This is followed by exploring GDM’s wide applications and fundamental principles and a comprehensive tutorial outlining the steps for using GDM in network optimization. In the context of intelligent networks, we study the impact of GDM on algorithms, e.g., **Deep Reinforcement Learning (DRL)**, and its implications for key scenarios, e.g., **incentive mechanism design**, **Semantic Communications(SemCom)**, **Internet of Vehicles (IoV) networks**, channel estimation, error correction coding, and channel denoising. We conclude our tutorial by discussing potential future research directions and summarizing the key contributions.

## ⚡ Network Optimization via Generative Diffusion Models
<img src="images/1.png" width = "100%">

GDM training approaches with and without an expert dataset. **Part A** illustrates the GDM training scenario when an expert database is accessible. The process learns from the GDM applications in the image domain: the optimal solution is retrieved from the expert database upon observing an environmental condition, followed by the GDM learning to replicate this optimal solution through forward diffusion and reverse denoising process. **Part B** presents the scenario where no expert database exists. In this case, GDM, with the assistance of a jointly trained solution evaluation network, learns to generate the optimal solution for a given environmental condition by actively exploring the unknown environment.

---

## 🔧 Environment Setup

To create a new conda environment, execute the following command:

```bash
conda create --name gdmopt python==3.8
```

## ⚡ Activate Environment

Activate the created environment with:

```bash
conda activate gdmopt
```

## 📦 Install Required Packages

The following package can be installed using pip:

```bash
pip install tianshou==0.4.11
```

## 🏃‍♀️ Run the Program



Run `main.py` in the file `Main` to start the program.

## 🔍 Check the results

---

## Citation

```bibtex
@article{du2023beyond,
  title={Beyond Deep Reinforcement Learning: A Tutorial on Generative Diffusion Models in Network Optimization},
  author={Authors},
  journal={},
  year={2023},
  publisher={IEEE}
}
```