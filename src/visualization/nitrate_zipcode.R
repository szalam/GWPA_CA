# Title: Plot Mean Nitrate Levels by Zip Code in California
# Description: This script reads in a dataset of nitrate levels at various locations in California,
#   calculates the mean nitrate level for each zip code, and plots a map of mean nitrate levels
#   by zip code using a US Census zip code shapefile.

rm(list=ls())

library(tidyverse)  # For data manipulation and plotting
library(tigris)     # For obtaining US Census data
library(sf)         # For working with spatial data
library(data.table)
library(htmlwidgets)
library(leaflet)
library(forcats) # For working with factor levels
library(dplyr)


# Set working directory and read in data
setwd('C:/sarfaraz/GWPA_dataset')
df = read.csv('nitrate_latlon.csv')[,-1]

# Remove rows with missing values
df = na.omit(df)

# Convert data to an sf object
data_sf = st_as_sf(df, coords = c('longitude','latitude'), crs = 4326)

# Get zip code shapefile and transform data to same CRS
zipshape = zctas(state = '06', year = '2010')

# Convert zipshape to a data frame for ggplot
zipshape_df <- st_set_geometry(zipshape, NULL)

# Create a data frame for the points
points_df <- as.data.frame(df)

# Transform data_sf to the same CRS as zipshape
data_sf_transformed <- st_transform(data_sf, crs = st_crs(zipshape))

# Spatially join the points to the zipshape polygons
joined_data <- st_join(zipshape, data_sf_transformed)

# Convert the joined_data to a data frame
joined_data_df <- as.data.frame(joined_data)

# Group the data by ZCTA5CE10 and calculate the mean of mean_nitrate and well count for each group
mean_nitrate_and_count_by_zip_allwell <- joined_data_df %>%
  group_by(ZCTA5CE10) %>%
  summarize(mean_nitrate = mean(mean_nitrate, na.rm = TRUE),
            well_count = n())

mean_nitrate_and_count_by_zip_well_type <- joined_data_df %>%
  group_by(ZCTA5CE10, well_type) %>%
  summarize(mean_nitrate = mean(mean_nitrate, na.rm = TRUE),
            well_count = n())

# Merge zipshape_df and mean_nitrate_and_count_by_zip based on ZCTA5CE10
merged_data_allwell <- left_join(zipshape, mean_nitrate_and_count_by_zip_allwell, by = "ZCTA5CE10")
merged_data_well_type <- left_join(zipshape, mean_nitrate_and_count_by_zip_well_type, by = "ZCTA5CE10")

# Plot the merged data
p = ggplot() +
  geom_sf(data = merged_data_allwell, aes(fill = mean_nitrate), color = NA) +
  geom_sf(data = merged_data_allwell, aes(fill = mean_nitrate), color = 'black', size = 0.01, linetype = "dashed") +
  coord_sf() +
  labs(title = "Mean Nitrate by Zip Code") +
  scale_fill_gradient2(low = "darkgreen", mid = "yellow", high = "red",
                       midpoint = 5, name = "Mean Nitrate",
                       na.value = "white", limits = c(0, 10),
                       guide = guide_colorbar(ticks.colour = "black", ticks.linewidth = 0.5)) +
  theme_minimal() +
  theme(
    axis.text = element_blank(),  # Remove axis text
    axis.ticks = element_blank(), # Remove axis ticks
    axis.title = element_blank(), # Remove axis titles
    plot.background = element_rect(fill = "white"), # Set plot background to white
    panel.grid = element_blank(), # Remove grid lines
    plot.title.position = "plot"  # Center the plot title
  ) +
  theme(plot.title = element_text(hjust = 0.5)) # Center alignment

p
# Save the ggplot image to a file (e.g., PNG format)
ggsave(filename = "C:/sarfaraz/GWPA_dataset/mean_nitrate_map_zipcode.png", plot = p, width = 10, height = 12, dpi = 300)



# Define breaks for the different groups
breaks <- c(0, 2, 4, 6, 8, 10, Inf)

# Define color palette for each group
colors <- rev(RColorBrewer::brewer.pal(length(breaks) - 1, "RdYlGn"))

# Define labels for each group
labels <- c("0-2", ">2-4", ">4-6", ">6-8", ">8-10", ">10")


# merged_data_sf = merged_data_allwell

# Calculate well counts for each ZCTA5CE10 and well_type
well_counts_by_zip <- joined_data_df %>%
  group_by(ZCTA5CE10, well_type) %>%
  summarize(well_count = n())

# Merge well counts with mean_nitrate_by_zip data
mean_nitrate_by_zip_well_type <- left_join(mean_nitrate_by_zip_well_type, well_counts_by_zip, by = c("ZCTA5CE10", "well_type"))

# Merge zipshape_df and mean_nitrate_by_zip_well_type based on ZCTA5CE10
merged_data_well_type <- left_join(zipshape, mean_nitrate_by_zip_well_type, by = "ZCTA5CE10")

# Modify process_merged_data() to include well_count
process_merged_data <- function(merged_data, well_type = NA) {
  
  if (!is.na(well_type)) {
    merged_data <- merged_data[merged_data$well_type == well_type,]
  }
  
  # Create a new column with group labels based on the 'mean_nitrate' column values
  merged_data$group <- cut(merged_data$mean_nitrate, breaks = breaks, labels = labels, include.lowest = TRUE, right = FALSE)
  
  # Calculate well count for each group
  merged_data <- merged_data %>%
    group_by(ZCTA5CE10) %>%
    mutate(well_count = n()) %>%
    ungroup()
  
  # Remove rows with NA values in the 'mean_nitrate' column
  merged_data <- merged_data[!is.na(merged_data$mean_nitrate), ]
  
  # Transform merged_data to the WGS 84 coordinate reference system
  merged_data_wgs84 <- st_transform(merged_data, crs = 4326)
  
  merged_data_wgs84$mean_nitrate = round(merged_data_wgs84$mean_nitrate, 2)
  
  merged_data_wgs84 = merged_data_wgs84[c('ZCTA5CE10', 'mean_nitrate', 'geometry', 'group', 'well_count')]
  
  return(merged_data_wgs84)
}


merged_data_sf_allwell <- process_merged_data(merged_data_allwell)
# Remove the well_count column from merged_data_sf_allwell
merged_data_sf_allwell <- select(merged_data_sf_allwell, -well_count)

# Convert merged_data_allwell to the same CRS as merged_data_sf_allwell
merged_data_allwell_wgs84 <- st_transform(merged_data_allwell, st_crs(merged_data_sf_allwell))

# Join the well_count column from merged_data_allwell_wgs84 based on matching ZCTA5CE10
merged_data_sf_allwell_updated <- st_join(merged_data_sf_allwell, 
                                          select(merged_data_allwell_wgs84, ZCTA5CE10, well_count),
                                          left = TRUE,join = st_equals)

# Remove the ZCTA5CE10_y column and rename ZCTA5CE10.x to ZCTA5CE10
merged_data_sf_allwell_updated <- select(merged_data_sf_allwell_updated, -ZCTA5CE10.y)
names(merged_data_sf_allwell_updated)[names(merged_data_sf_allwell_updated) == "ZCTA5CE10.x"] <- "ZCTA5CE10"


merged_data_sf_domestic <- process_merged_data(merged_data_well_type, well_type = 'Domestic')
# Remove the well_count column from merged_data_sf_allwell
merged_data_sf_domestic <- select(merged_data_sf_domestic, -well_count)

# Convert merged_data_allwell to the same CRS as merged_data_sf_allwell
merged_data_sf_domestic_wgs84 <- st_transform(merged_data_well_type, st_crs(merged_data_sf_domestic))

# Join the well_count column from merged_data_allwell_wgs84 based on matching ZCTA5CE10
merged_data_sf_domestic_updated <- st_join(merged_data_sf_domestic, 
                                          select(merged_data_sf_domestic_wgs84, ZCTA5CE10, well_count),
                                          left = TRUE,join = st_equals)
# Remove the ZCTA5CE10_y column and rename ZCTA5CE10.x to ZCTA5CE10
merged_data_sf_domestic_updated <- select(merged_data_sf_domestic_updated, -ZCTA5CE10.y)
names(merged_data_sf_domestic_updated)[names(merged_data_sf_domestic_updated) == "ZCTA5CE10.x"] <- "ZCTA5CE10"



well_count_palette <- colorNumeric(palette = "magma", domain = merged_data_sf_allwell_updated$well_count, na.color = "gray")


mean_nitrate_map <- leaflet() %>%
  addProviderTiles(provider = basemap, group = "Basemap") %>%
  addPolygons(data = merged_data_sf_allwell_updated,
              fillColor = ~colorFactor(colors, levels = labels, na.color = "gray")(group),
              color = "black", weight = 0.5, 
              fillOpacity = 1,
              label = ~paste0("Zip Code: ", ZCTA5CE10, "<br>Mean Nitrate (All): ", mean_nitrate, "<br>Well Count (All): ", well_count),
              group = "Mean Nitrate (All)") %>%
  addPolygons(data = merged_data_sf_domestic_updated,
              fillColor = ~colorFactor(colors, levels = labels, na.color = "gray")(group),
              color = "black", weight = 0.5, 
              fillOpacity = 1,
              label = ~paste0("Zip Code: ", ZCTA5CE10, "<br>Mean Nitrate (Domestic): ", mean_nitrate, "<br>Well Count (Domestic): ", well_count),
              group = "Mean Nitrate (Domestic)") %>%
  addLegend("bottomright", title = "Mean Nitrate [mg/l]",
            colors = colors, labels = labels, opacity = 1) %>%
  addLayersControl(
    overlayGroups = c("Mean Nitrate (All)", "Mean Nitrate (Domestic)"),
    options = layersControlOptions(collapsed = FALSE)
  ) %>%
  setView(lng = mean(st_coordinates(merged_data_sf_allwell)[,1]), # Center the map based on data
          lat = mean(st_coordinates(merged_data_sf_allwell)[,2]),
          zoom = 7) # Zoom level

mean_nitrate_map

saveWidget(mean_nitrate_map, file = "C:/sarfaraz/GWPA_dataset/well_nitrate_map_zipcode.html", selfcontained = FALSE)

well_count_map <- leaflet() %>%
  addProviderTiles(provider = basemap, group = "Basemap") %>%
  addPolygons(data = merged_data_sf_allwell_updated,
              fillColor = ~well_count_palette(well_count),
              color = "black", weight = 0.5, 
              fillOpacity = 1,
              label = ~paste0("Zip Code: ", ZCTA5CE10, "<br>Well Count (All): ", well_count),
              group = "Well Count (All)") %>%
  addPolygons(data = merged_data_sf_domestic_updated,
              fillColor = ~well_count_palette(well_count),
              color = "black", weight = 0.5, 
              fillOpacity = 1,
              label = ~paste0("Zip Code: ", ZCTA5CE10, "<br>Well Count (Domestic): ", well_count),
              group = "Well Count (Domestic)") %>%
  addLegend("bottomright", title = "Well Count",
            pal = well_count_palette, values = merged_data_sf_allwell_updated$well_count, opacity = 1) %>%
  addLayersControl(
    overlayGroups = c( "Well Count (All)", "Well Count (Domestic)"),
    options = layersControlOptions(collapsed = FALSE)
  ) %>%
  setView(lng = mean(st_coordinates(merged_data_sf_allwell)[,1]), # Center the map based on data
          lat = mean(st_coordinates(merged_data_sf_allwell)[,2]),
          zoom = 7) # Zoom level

well_count_map

saveWidget(well_count_map, file = "C:/sarfaraz/GWPA_dataset/well_count_map_zipcode.html", selfcontained = FALSE)
