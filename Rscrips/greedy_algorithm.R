train = read.csv('/Users/pfournel/projetWeb/projectB/input/gifts.csv')
best_sol = 1000000000000000
for(number_of_clusters in 210:400){
  fit = kmeans(train$Longitude, number_of_clusters)

  #Creating dataframe
  res = train[,c(1,2,3,4,1,1)]
  colnames(res) = c("GiftId", "Latitude", "Longitude", "Weight", "TripId", "Cluster")
  res$TripId = 0
  res$Cluster = fit$cluster

  tour = 0
  for(cluster in 1:length(fit$centers)){
    items_in_cluster = train[fit$cluster == cluster,]
    items_in_cluster = items_in_cluster[order(items_in_cluster$Latitude),]
    cum_sum_weight = cumsum(items_in_cluster$Weight)
    range = 600:950
    sleigh_weight = range[which.min(unlist(lapply(tail(cum_sum_weight, 1) / range, function(x) ceiling(x) - x)))]
    cuts = unlist(lapply(cum_sum_weight, function(x) { return(ceiling(x / sleigh_weight)) })) + tour
    res[items_in_cluster$GiftId, ]$TripId = cuts
    tour = tail(cuts, 1)
  }
  final = res[order(res$Latitude, decreasing = TRUE), ]

  file_name = "temp_sol.csv"
  write.table(final[,c(1,5)], file = paste0("/Users/pfournel/projetWeb/projectB/solutions/", file_name), sep = ",",row.names=FALSE)
  val = as.numeric(system(paste0("python /Users/pfournel/projetWeb/projectB/scripts/check_sol.py ",file_name), intern=TRUE))
  if(val<best_sol){
    best_sol = val
    print(val)
    write.table(final[,c(1,5)], file = paste0("/Users/pfournel/projetWeb/projectB/solutions/best_sol_",number_of_clusters,"_", val, ".csv"), sep = ",",row.names=FALSE)
  }
}