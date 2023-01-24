library(rgdal)


soil_loc = '/Volumes/GoogleDrive/Shared drives/GWAttribution/data/raw/soil/soils/mlra_a_ca.shp'
soldata = readOGR(soil_loc)
plot(soldata)

# The input file geodatabase
fgdb = '/Volumes/GoogleDrive/Shared drives/GWAttribution/data/raw/soil/gSSURGO_CA.gdb'

# List all feature classes in a file geodatabase
subset(ogrDrivers(), grepl("GDB", name))
fc_list <- ogrListLayers(fgdb)
print(fc_list)

# Read the feature class
fc <- readOGR(dsn=fgdb,layer="fras_aux_MapunitRaster_10m_CA_202210")

# Determine the FC extent, projection, and attribute information
summary(fc)

# View the feature class
plot(fc[1:50000,])
head(fc)
