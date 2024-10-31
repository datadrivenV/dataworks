// Please run the 'use Fandango;' cmd before running this file as 'use' function is not valid in .js file

// Insert data into the Parkar collection
db.createCollection("Parkar");

db.Parkar.insertMany([
  {
    "theaterName": "Cinemark Tinseltown Rochester",
    "location": "2609 West Henrietta Road, Rochester, NY 14623",
    "movies": [
      {
        "movieTitle": "Afraid",
        "genre": "Horror, Mystery & Thriller",
        "rating": "PG-13",
        "duration": 84,
        "showtimes": {
          "2024-08-29": ["7:40 PM"],
          "2024-08-30": ["4:00 PM", "7:15 PM", "9:45 PM"]
        },
        "features": ["Closed caption", "Recliner seats", "Accessibility devices"]
      },
      {
        "movieTitle": "Alien: Romulus",
        "genre": "Horror, Sci-Fi/Fantasy",
        "rating": "R",
        "duration": 119,
        "showtimes": {
          "2024-08-29": ["2:50 PM", "5:30 PM"],
          "2024-08-30": ["4:30 PM", "7:10 PM"]
        },
        "features": ["Closed caption", "Recliner seats", "Accessibility devices"]
      },
      {
        "movieTitle": "The Crow",
        "genre": "Action/Adventure, Sci-Fi/Fantasy",
        "rating": "R",
        "duration": 111,
        "showtimes": {
          "2024-08-29": ["3:00 PM", "5:40 PM", "8:20 PM"],
          "2024-08-30": ["4:00 PM", "7:15 PM"]
        },
        "features": ["Closed caption", "Recliner seats", "Accessibility devices"]
      }
    ]
  },
  {
    "theaterName": "Apple Cinemas Pittsford",
    "location": "3349 Monroe Avenue, Pittsford, NY 14534",
    "movies": [
      {
        "movieTitle": "Afraid",
        "genre": "Horror, Mystery & Thriller",
        "rating": "PG-13",
        "duration": 84,
        "showtimes": {
          "2024-08-29": ["4:00 PM", "7:30 PM"],
          "2024-08-30": ["6:30 PM", "9:00 PM"]
        },
        "features": ["Reserved seating", "Recliner seats"]
      },
      {
        "movieTitle": "It Ends With Us",
        "genre": "Drama, Romance",
        "rating": "PG-13",
        "duration": 130,
        "showtimes": {
          "2024-08-29": ["4:20 PM", "7:20 PM"],
          "2024-08-30": ["4:30 PM", "8:00 PM"]
        },
        "features": ["Reserved seating", "Recliner seats"]
      },
      {
        "movieTitle": "Deadpool & Wolverine",
        "genre": "Action/Adventure, Comedy",
        "rating": "R",
        "duration": 127,
        "showtimes": {
          "2024-08-29": ["4:00 PM", "7:30 PM"],
          "2024-08-30": ["5:10 PM", "8:00 PM"]
        },
        "features": ["Reserved seating", "Recliner seats"]
      }
    ]
  }
]);


/* Print all documents in the Parkar collection as an array
var theaters = db.Parkar.find().toArray();
printjson(theaters); */

//used custom print function for more readability
print("===============================================");
print("            Movie Times + Tickets              ");
print("               near Rochester                 ");
print("===============================================");

//Function to display a formatted list of theatres and movies
db.Parkar.find().forEach(function (theater) {
  print("Theater: " + theater.theaterName);
  print("Location: " + theater.location);
  print("Movies:");
  theater.movies.forEach(function (movie) {
    print("  Movie Title: " + movie.movieTitle);
    print("  Genre: " + movie.genre);
    print("  Rating: " + movie.rating);
    print("  Duration: " + movie.duration + " minutes");
    print("  Showtimes:");
    for (var date in movie.showtimes) {
      print("    " + date + ": " + movie.showtimes[date].join(", "));
    }
    print("  Features: " + movie.features.join(", "));
    print("");
  });
  print("--------------------------------------------------");
}); 
