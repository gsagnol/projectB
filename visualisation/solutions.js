var fs = require('fs');
var parse = require('csv-parse');
var haversine = require('haversine');

var cachedSolutions = {};

var poleNord = {
    latitude: 90,
    longitude: 0
};

Array.prototype.unique = function() {
    var a = [];
    for (var i=0, l=this.length; i<l; i++)
        if (a.indexOf(this[i]) === -1)
            a.push(this[i]);
    return a;
};
var inputData;

fs.readFile('../input/gifts.csv', "utf-8", function(err, data) {
    parse(data, {columns:["GiftId","Latitude","Longitude","Weight"]}, function(err, output){
        output.shift();
        output = output.reduce(function(map, obj) {
            var giftId = parseInt(obj["GiftId"]);
            var lat = parseFloat(obj["Latitude"]);
            var lon = parseFloat(obj["Longitude"]);
            var weight = parseFloat(obj["Weight"]);
            map[giftId] = {"Latitude": lat, "Longitude": lon, "Weight":weight};

            return map;
        }, {});
        inputData = output;
    });
});

var loadFile = function(solution, callback){
    if (solution in cachedSolutions){
        console.log("Using cached data.");
        callback(cachedSolutions[solution]);
    }else{
        fs.readFile('../solutions/'+solution, "utf-8", function(err, data) {
            parse(data, {columns:["GiftId", "TripId"]}, function(err, output){
                output.shift();
                output = output.map(function(obj){
                    var giftId = parseInt(obj["GiftId"]);
                    var tripId = parseInt(obj["TripId"]);
                    var lat = inputData[giftId]["Latitude"];
                    var lon = inputData[giftId]["Longitude"];
                    var weight = inputData[giftId]["Weight"];
                    return {"GiftId": giftId, "TripId": tripId, "Latitude": lat, "Longitude": lon, "Weight":weight};
                });
                cachedSolutions[solution] = output;
                callback(cachedSolutions[solution]);
            });
        });
    }
};


module.exports = {
    getNames: function(callback){
        fs.readdir("../solutions", function(err, items) {
            callback(items);
        });
    },
    getTrips: function(solution, callback){
        loadFile(solution, function(solutionData){
            var trips = solutionData.map(function(x){
                return x["TripId"];
            });
            callback(trips.unique().sort(function(a, b) {return a - b;}));
        })
    },
    getTripsInArea: function(solution, area, callback){
        loadFile(solution, function(solutionData){
            var trips = solutionData.filter(function(x){
                return x["Longitude"] < area["maxLon"] && x["Longitude"] > area["minLon"] && x["Latitude"] < area["maxLat"] && x["Latitude"] > area["minLat"]
            }).map(function(x){
                return x["TripId"];
            });
            callback(trips.unique().sort(function(a, b) {return a - b;}));
        })
    },
    getWorseTrips: function(solution, callback){
        loadFile(solution, function(solutionData){
            var trips = {};
            var tripIds = [];
            var distTotal = 0;
            solutionData.forEach(function(obj){
                if(obj["TripId"] in trips){
                    trips[obj["TripId"]]["Gifts"].push(obj);
                }else{
                    trips[obj["TripId"]] = {"TripId": obj["TripId"], "Cost": 0, "TotalWeight": 0, "Gifts": [obj]};
                    tripIds.push(obj["TripId"]);
                }
            });
            var tripArray = [];
            tripIds.forEach(function(tripId){
                var sumTrips = 10;
                var borne = 0;
                for(var i=0; i < trips[tripId]["Gifts"].length;i++){
                    sumTrips += trips[tripId]["Gifts"][i]["Weight"];
                    var current = {
                        latitude: trips[tripId]["Gifts"][i]["Latitude"],
                        longitude: trips[tripId]["Gifts"][i]["Longitude"]
                    };
                    borne += haversine(poleNord, current) * trips[tripId]["Gifts"][i]["Weight"];
                }

                borne = 1.02 * borne;
                var dist = 0;
                var previous = poleNord;
                var totalWeight = 0;
                trips[tripId]["Gifts"].push({"Latitude": poleNord.latitude, "Longitude":poleNord.longitude, "Weight": 0});
                for(var i=0; i < trips[tripId]["Gifts"].length; i++){
                    var current = {
                        latitude: trips[tripId]["Gifts"][i]["Latitude"],
                        longitude: trips[tripId]["Gifts"][i]["Longitude"]
                    };

                    dist = dist + haversine(previous, current) * sumTrips;
                    sumTrips = sumTrips - trips[tripId]["Gifts"][i]["Weight"];
                    previous = current;

                    totalWeight+= trips[tripId]["Gifts"][i]["Weight"];
                }
                trips[tripId]["Cost"] = dist / borne ;
                trips[tripId]["TotalWeight"] = totalWeight ;
                tripArray.push(trips[tripId]);
                distTotal += dist;
            });
            console.log(distTotal);
            var res = tripArray.sort(function(a, b){ return b["Cost"] - a["Cost"]; }).map(function(x){
                var tripIdI = x["TripId"];
                return x["TripId"];
            });
            callback(res.splice(0,Math.floor(res.length * 0.10)));
        })
    },
    getTrip: function(solution, tripId, callback){
        loadFile(solution, function(solutionData){
            var trip = solutionData.filter(function(x){
                return x["TripId"] == tripId;
            });
            callback(trip);
        })
    }
};