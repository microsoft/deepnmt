[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Very Deep Transformers for Neural Machine Translation

This PyTorch package implements Very Deep Transformers for Neural Machine Translation, as described in:

Xiaodong Liu,  Kevin Duh, Liyuan Liu and Jianfeng Gao<br/>
Very deep transformers for neural machine translation <br/>
[arXiv version](https://arxiv.org/abs/2008.07772) <br/>

## Quickstart

### Model Training
1. Data Preprocessing
   Please follow instructions: (https://github.com/pytorch/fairseq/tree/main/examples/scaling_nmt)
2. Model Train </br>
   > bash run_wmt14_en_fr.sh
3. Model Eval </br>
   > cd nmt_eval && bash eval_enfr.sh <model_path> <gpu> <init_path> <unperbound> <count>

## Notes and Acknowledgments
FAIRSEQ: https://github.com/pytorch/fairseq<br/>


### How do I cite it?

```

@article{liu2020deepnmt,
  title={Very deep transformers for neural machine translation},
  author={Liu, Xiaodong and Duh, Kevin and Liu, Liyuan and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2008.07772},
  year={2020}
}
```
### Contact Information

For personal communication related to this package, please contact Xiaodong Liu (`xiaodl@microsoft.com`).
  

> This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
