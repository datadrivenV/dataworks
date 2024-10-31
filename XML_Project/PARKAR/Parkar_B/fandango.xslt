<?xml version="1.0"?> 
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">

  <xsl:template match="/movies">
    <html>
      <head>
        <title>Movie Listings</title>
      </head>
      <body>
        <h1>Movie Listings</h1>
        <xsl:for-each select="theater">
          <h2>Theater: <xsl:value-of select="@name"/></h2>
          <table border="1">
            <tr>
              <th>Title</th>
              <th>Genre</th>
              <th>Duration</th>
              <th>Showtimes</th>
              <th>Seating Info</th>
              <th>Image</th>
            </tr>
            <xsl:for-each select="movie">
              <tr>
                <td><xsl:value-of select="@title"/></td>
                <td><xsl:value-of select="genre"/></td>
                <td><xsl:value-of select="duration"/></td>
                <td>
                  <xsl:for-each select="showtimes/showtime">
                    <xsl:value-of select="."/>
                    <br/>
                  </xsl:for-each>
                </td>
                <td><xsl:value-of select="seating"/></td>
                <td>
                  <img src="{image}" alt="{@title} poster" width="100" height="150"/>
                </td>
              </tr>
            </xsl:for-each>
          </table>
        </xsl:for-each>
      </body>
    </html>
  </xsl:template>
  
</xsl:stylesheet>
