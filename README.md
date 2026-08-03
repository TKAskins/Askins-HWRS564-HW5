# Askins-UoA-Research
Repository created for Master Thesis Reseach by Trevor Askins, UoA class of 2026
Efficient irrigation management requires balancing adequate root-zone water availability with the prevention of excessive deep percolation, nutrient loss, and unnecessary water consumption. Currently, site specific practices rely on interpolation of previous scientific studies and lessons learned. Manually creating site-specific models to guide sensor network design would be prohibitively time consuming due to the sheer number of potential sensor combinations. This creates a gap between existing crop data and site-specific applications. This study evaluates cost-aware decision tree regression as a methodology for optimizing soil sensor placement in irrigated agricultural systems. HYDRUS-1D simulations were used to generate synthetic datasets representing pressure head, volumetric water content, and water flux for 45 representative agricultural soils cultivated with corn and beans under flood irrigation. Variable root depths throughout the growing season were incorporated to produce representative training data for machine-learning models. Decision trees were trained to predict deep percolation, average root-zone water content, and a combined monitoring objective. A modified cost-aware decision tree algorithm was then used to balance predictive accuracy against sensor installation, equipment, and measurement costs. Results indicate that moderate cost weighting frequently improved prediction accuracy while substantially reducing the number and overall cost of required sensors, suggesting that cost weighting acts as an effective regularization strategy. The optimized monitoring networks reduced deployment costs while maintaining high predictive performance, with root-zone water-content predictions achieving errors below those commonly associated with high-quality field measurements. Sensor selection varied with crop type and monitoring objective, indicating that no universally optimal sensor layout exists across agricultural settings. The proposed methodology provides a practical framework for designing cost-effective irrigation monitoring networks and demonstrates the potential of integrating numerical vadose zone modeling with machine-learning techniques to support precision irrigation.

To Use This Repository:
1) The flux will be modeled for multiple crops under multiple irrigation methods
2) The parameters relevant to prediccting that flux will be associated with a soil sensor
3) Soil sensors will be linked to a cost function
4) ML will identify the best set of sensors to macximize soil monitoring accuracy while minimizing costs

Crops: The crops that will be modeled will be corn and wheat as they are two of the top three crops in worldwide production
Irrigation Methods: Flood irrigation, sprinkler irrigation, drip  irrigation
Soil Type: TBD
Cost functions: TBD

# Kernel/Enviroment
The packages used to run this research will be listed here with their associated install codes (my local version is still being refined, once it is more stable I may add the kernel directly to the enviroment):
1) 
