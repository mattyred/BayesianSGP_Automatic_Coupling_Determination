
- Check if sghmc is convergin when using BGP
- Add WILT, DIABETS datasets
- Keep normalization of full dataset with classification, ...why?
- Check BGP regression is still working with new conditional
- Check BGP classification on the two smallest datasets
- Change MNLL for classification

## Future (minor) improvments
- Create a common gpmodel class since lots of methods are shared between BSGP and BGP
- Put in some utils file clip_grad function and jacobian function
- Change in kernels (ACD) when using torch.zeros and torch.ones device=... to be device=self.device (thus define a device for the entire kernel), maybe doing kernel.to(device) is enough?
- Change metric computations just to use pytorch
- Create a parent class for BGP and BSGP