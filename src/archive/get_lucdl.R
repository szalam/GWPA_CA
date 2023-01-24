# Required Libraries
library(raster)
library(sf)

# Read TIF file
land_use_tif <- raster("/Volumes/GoogleDrive/Shared drives/GWAttribution/data/raw/USDA_CDL/CDL_2021_clip_20230113134140_113242721.tif.")

# Read shapefile
wells_shp <- st_read("/Volumes/GoogleDrive/Shared drives/GWAttribution/data/processed/Well_buffer_shape/UCD_buffers_2mile.shp")

#Transform to TIF's Coordinate Reference System (CRS)
wells_shp_crs <- st_transform(wells_shp, crs(land_use_tif))

# Extract land use values
wells_land_use <- extract(land_use_tif, wells_shp_crs)

# Crop land use values
crop_land_use <- wells_land_use[wells_land_use %in% c(1,2,3)]

# Water land use values
water_land_use <- wells_land_use[wells_land_use %in% c(5,6,8)]

# Calculate area of crop and water land use
crop_area <- length(crop_land_use) * 900 # 900 sq meter (30*30)
water_area <- length(water_land_use) * 900 # 900 sq meter (30*30)

# Create dataframe
well_area_df <- data.frame(well_id = wells_shp$well_id,
                           crop_area = crop_area,
                           water_area = water_area)

# Export dataframe to CSV
write.csv(well_area_df, "well_area.csv", row.names = FALSE)
