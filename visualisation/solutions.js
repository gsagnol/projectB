var fs = require('fs');
var parse = require('csv-parse');

var cachedSolutions = {};

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
            var weight = parseInt(obj["Weight"]);
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
    getTrip: function(solution, tripId, callback){
        loadFile(solution, function(solutionData){
            var trip = solutionData.filter(function(x){
                return x["TripId"] == tripId;
            });
            callback(trip);
        })
    }
};