/*
    D3.js Slider
    Inspired by jQuery UI Slider
    Copyright (c) 2013, Bjorn Sandvik - http://blog.thematicmapping.org
    BSD license: http://opensource.org/licenses/BSD-3-Clause
*/
(function (root, factory) {
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define(['d3'], factory);
  } else if (typeof exports === 'object') {
    if (process.browser) {
      // Browserify. Import css too using cssify.
      require('./d3.slider.css');
    }
    // Node. Does not work with strict CommonJS, but
    // only CommonJS-like environments that support module.exports,
    // like Node.
    module.exports = factory(require('d3'));
  } else {
    // Browser globals (root is window)
    root.d3.slider = factory(root.d3);
  }
}(this, function (d3) {
return function module() {

  "use strict";

  // Public variables width default settings
  var min = 0,
      max = 100,
      step = 0.01,
      animate = true,
      orientation = "horizontal",
      axis = false,
      margin = 100,
      value,
      active = 1,
      snap = false,
      scale;
	  
var alldates = ['2000-01-01','2000-02-01','2000-03-01','2000-04-01','2000-05-01','2000-06-01','2000-07-01','2000-08-01','2000-09-01','2000-10-01','2000-11-01','2000-12-01','2001-01-01','2001-02-01','2001-03-01','2001-04-01','2001-05-01','2001-06-01','2001-07-01','2001-08-01','2001-09-01','2001-10-01','2001-11-01','2001-12-01','2002-01-01','2002-02-01','2002-03-01','2002-04-01','2002-05-01','2002-06-01','2002-07-01','2002-08-01','2002-09-01','2002-10-01','2002-11-01','2002-12-01','2003-01-01','2003-02-01','2003-03-01','2003-04-01','2003-05-01','2003-06-01','2003-07-01','2003-08-01','2003-09-01','2003-10-01','2003-11-01','2003-12-01','2004-01-01','2004-02-01','2004-03-01','2004-04-01','2004-05-01','2004-06-01','2004-07-01','2004-08-01','2004-09-01','2004-10-01','2004-11-01','2004-12-01','2005-01-01','2005-02-01','2005-03-01','2005-04-01','2005-05-01','2005-06-01','2005-07-01','2005-08-01','2005-09-01','2005-10-01','2005-11-01','2005-12-01','2006-01-01','2006-02-01','2006-03-01','2006-04-01','2006-05-01','2006-06-01','2006-07-01','2006-08-01','2006-09-01','2006-10-01','2006-11-01','2006-12-01','2007-01-01','2007-02-01','2007-03-01','2007-04-01','2007-05-01','2007-06-01','2007-07-01','2007-08-01','2007-09-01','2007-10-01','2007-11-01','2007-12-01','2008-01-01','2008-02-01','2008-03-01','2008-04-01','2008-05-01','2008-06-01','2008-07-01','2008-08-01','2008-09-01','2008-10-01','2008-11-01','2008-12-01','2009-01-01','2009-02-01','2009-03-01','2009-04-01','2009-05-01','2009-06-01','2009-07-01','2009-08-01','2009-09-01','2009-10-01','2009-11-01','2009-12-01','2010-01-01','2010-02-01','2010-03-01','2010-04-01','2010-05-01','2010-06-01','2010-07-01','2010-08-01','2010-09-01','2010-10-01','2010-11-01','2010-12-01','2011-01-01','2011-02-01','2011-03-01','2011-04-01','2011-05-01','2011-06-01','2011-07-01','2011-08-01','2011-09-01','2011-10-01','2011-11-01','2011-12-01','2012-01-01','2012-02-01','2012-03-01','2012-04-01','2012-05-01','2012-06-01','2012-07-01','2012-08-01','2012-09-01','2012-10-01','2012-11-01','2012-12-01','2013-01-01','2013-02-01','2013-03-01','2013-04-01','2013-05-01','2013-06-01','2013-07-01','2013-08-01','2013-09-01','2013-10-01','2013-11-01','2013-12-01','2014-01-01','2014-02-01','2014-03-01','2014-04-01','2014-05-01','2014-06-01','2014-07-01','2014-08-01','2014-09-01','2014-10-01','2014-11-01','2014-12-01','2015-01-01','2015-02-01','2015-03-01','2015-04-01','2015-05-01','2015-06-01','2015-07-01','2015-08-01','2015-09-01','2015-10-01','2015-11-01','2015-12-01','2016-01-01','2016-02-01','2016-03-01','2016-04-01','2016-05-01']

  // Private variables
  var axisScale,
      dispatch = d3.dispatch("slide", "slideend"),
      formatPercent = d3.format(".2%"),
      tickFormat = d3.format(".0"),
      handle1,
      handle2 = null,
      divRange,
      sliderLength;

  function slider(selection) {
    selection.each(function() {

      // Create scale if not defined by user
      if (!scale) {
        scale = d3.scale.linear().domain([min, max]);
      }

      // Start value
      value = value || scale.domain()[0];

      // DIV container
      var div = d3.select(this).classed("d3-slider d3-slider-" + orientation, true);
      
      var drag = d3.behavior.drag();
      drag.on('dragend', function () {
        dispatch.slideend(d3.event, value);
      })

      // Slider handle
      //if range slider, create two
      // var divRange;

      if (toType(value) == "array" && value.length == 2) {
        handle1 = div.append("a")
          .classed("d3-slider-handle", true)
          .attr("xlink:href", "#")
          .attr('id', "handle-one")
          .on("click", stopPropagation)
          .call(drag);
        handle2 = div.append("a")
          .classed("d3-slider-handle", true)
          .attr('id', "handle-two")
          .attr("xlink:href", "#")
          .on("click", stopPropagation)
          .call(drag);
      } else {
        handle1 = div.append("a")
          .classed("d3-slider-handle", true)
          .attr("xlink:href", "#")
          .attr('id', "handle-one")
          .on("click", stopPropagation)
          .call(drag);
      }
      
      // Horizontal slider
      if (orientation === "horizontal") {

        div.on("click", onClickHorizontal);
        
        if (toType(value) == "array" && value.length == 2) {
          divRange = d3.select(this).append('div').classed("d3-slider-range", true);

          handle1.style("left", formatPercent(scale(value[ 0 ])));
          divRange.style("left", formatPercent(scale(value[ 0 ])));
          drag.on("drag", onDragHorizontal);

          var width = 100 - parseFloat(formatPercent(scale(value[ 1 ])));
          handle2.style("left", formatPercent(scale(value[ 1 ])));
          divRange.style("right", width+"%");
          drag.on("drag", onDragHorizontal);

        } else {
          handle1.style("left", formatPercent(scale(value)));
          drag.on("drag", onDragHorizontal);
        }
        
        sliderLength = parseInt(div.style("width"), 10);

      } else { // Vertical

        div.on("click", onClickVertical);
        drag.on("drag", onDragVertical);
        if (toType(value) == "array" && value.length == 2) {
          divRange = d3.select(this).append('div').classed("d3-slider-range-vertical", true);

          handle1.style("bottom", formatPercent(scale(value[ 0 ])));
          divRange.style("bottom", formatPercent(scale(value[ 0 ])));
          drag.on("drag", onDragVertical);

          var top = 100 - parseFloat(formatPercent(scale(value[ 1 ])));
          handle2.style("bottom", formatPercent(scale(value[ 1 ])));
          divRange.style("top", top+"%");
          drag.on("drag", onDragVertical);

        } else {
          handle1.style("bottom", formatPercent(scale(value)));
          drag.on("drag", onDragVertical);
        }
        
        sliderLength = parseInt(div.style("height"), 10);

      }
      
      if (axis) {
        createAxis(div);
      }


      function createAxis(dom) {

        // Create axis if not defined by user
        if (typeof axis === "boolean") {
	
          axis = d3.svg.axis()
              //.ticks(Math.round(sliderLength / 100)) --original code
			  .ticks(19) /*changed to show 20 years instead*/
              .tickFormat(tickFormat)
              .orient((orientation === "horizontal") ? "bottom" :  "right");

        }

        // Copy slider scale to move from percentages to pixels
        axisScale = scale.ticks ? scale.copy().range([0, sliderLength]) : scale.copy().rangePoints([0, sliderLength], 0.5);
          axis.scale(axisScale);

          // Create SVG axis container
        var svg = dom.append("svg")
            .classed("d3-slider-axis d3-slider-axis-" + axis.orient(), true)
            .on("click", stopPropagation);

        var g = svg.append("g");

        // Horizontal axis
        if (orientation === "horizontal") {

          svg.style("margin-left", -margin + "px");

          svg.attr({
            width: sliderLength + margin * 2,
            height: margin
          });
/* --- changed to show predicted ---*/
		var pred = svg.append("g")
		 .append("rect")
		 .attr("class", "bar")
    .attr("y", (-10))
	.attr("height", 20)
	.attr("x", (margin + sliderLength*(198/216)))
	.attr("width", (sliderLength*(1 - (198/216))))
	.attr("fill", "red")
	.attr("opacity", .1);
	/*----*/
		
          if (axis.orient() === "top") {
            svg.style("top", -margin + "px");
            g.attr("transform", "translate(" + margin + "," + margin + ")");
          } else { // bottom
            g.attr("transform", "translate(" + margin + ",0)");
          }

        } else { // Vertical

          svg.style("top", -margin + "px");

          svg.attr({
            width: margin,
            height: sliderLength + margin * 2
          });

          if (axis.orient() === "left") {
            svg.style("left", -margin + "px");
            g.attr("transform", "translate(" + margin + "," + margin + ")");
          } else { // right          
            g.attr("transform", "translate(" + 0 + "," + margin + ")");
          }

        }

        g.call(axis);

      }

      function onClickHorizontal() {
        if (toType(value) != "array") {
          var pos = Math.max(0, Math.min(sliderLength, d3.event.offsetX || d3.event.layerX));
          moveHandle(scale.invert ? 
                      stepValue(scale.invert(pos / sliderLength))
                    : nearestTick(pos / sliderLength));
        }
      }

      function onClickVertical() {
        if (toType(value) != "array") {
          var pos = sliderLength - Math.max(0, Math.min(sliderLength, d3.event.offsetY || d3.event.layerY));
          moveHandle(scale.invert ? 
                      stepValue(scale.invert(pos / sliderLength))
                    : nearestTick(pos / sliderLength));
        }
      }

      function onDragHorizontal() {
        if ( d3.event.sourceEvent.target.id === "handle-one") {
          active = 1;
        } else if ( d3.event.sourceEvent.target.id == "handle-two" ) {
          active = 2;
        }
        var pos = Math.max(0, Math.min(sliderLength, d3.event.x));
        moveHandle(scale.invert ? 
                    stepValue(scale.invert(pos / sliderLength))
                  : nearestTick(pos / sliderLength));
      }

      function onDragVertical() {
        if ( d3.event.sourceEvent.target.id === "handle-one") {
          active = 1;
        } else if ( d3.event.sourceEvent.target.id == "handle-two" ) {
          active = 2;
        }
        var pos = sliderLength - Math.max(0, Math.min(sliderLength, d3.event.y))
        moveHandle(scale.invert ? 
                    stepValue(scale.invert(pos / sliderLength))
                  : nearestTick(pos / sliderLength));
      }

      function stopPropagation() {
        d3.event.stopPropagation();
      }

    });

  }

  // Move slider handle on click/drag
  function moveHandle(newValue) {
    var currentValue = toType(value) == "array"  && value.length == 2 ? value[active - 1]: value,
        oldPos = formatPercent(scale(stepValue(currentValue))),
        newPos = formatPercent(scale(stepValue(newValue))),
        position = (orientation === "horizontal") ? "left" : "bottom";
    if (oldPos !== newPos) {

      if (toType(value) == "array" && value.length == 2) {
        value[ active - 1 ] = newValue;
        if (d3.event) {
          dispatch.slide(d3.event, value );
        };
      } else {
        if (d3.event) {
          dispatch.slide(d3.event.sourceEvent || d3.event, value = newValue);
        };
      }

      if ( value[ 0 ] >= value[ 1 ] ) return;
      if ( active === 1 ) {
        if (toType(value) == "array" && value.length == 2) {
          (position === "left") ? divRange.style("left", newPos) : divRange.style("bottom", newPos);
        }

        if (animate) {
          handle1.transition()
              .styleTween(position, function() { return d3.interpolate(oldPos, newPos); })
              .duration((typeof animate === "number") ? animate : 250);
        } else {
          handle1.style(position, newPos);
        }
      } else {
        
        var width = 100 - parseFloat(newPos);
        var top = 100 - parseFloat(newPos);

        (position === "left") ? divRange.style("right", width + "%") : divRange.style("top", top + "%");
        
        if (animate) {
          handle2.transition()
              .styleTween(position, function() { return d3.interpolate(oldPos, newPos); })
              .duration((typeof animate === "number") ? animate : 250);
        } else {
          handle2.style(position, newPos);
        }
      }
    }
  }

  // Calculate nearest step value
  function stepValue(val) {

    if (val === scale.domain()[0] || val === scale.domain()[1]) {
      return val;
    }

    var alignValue = val;
    if (snap) {
      alignValue = nearestTick(scale(val));
    } else{
      var valModStep = (val - scale.domain()[0]) % step;
      alignValue = val - valModStep;

      if (Math.abs(valModStep) * 2 >= step) {
        alignValue += (valModStep > 0) ? step : -step;
      }
    };

    return alignValue;

  }

  // Find the nearest tick
  function nearestTick(pos) {
    var ticks = scale.ticks ? scale.ticks() : scale.domain();
    var dist = ticks.map(function(d) {return pos - scale(d);});
    var i = -1,
        index = 0,
        r = scale.ticks ? scale.range()[1] : scale.rangeExtent()[1];
    do {
        i++;
        if (Math.abs(dist[i]) < r) {
          r = Math.abs(dist[i]);
          index = i;
        };
    } while (dist[i] > 0 && i < dist.length - 1);

    return ticks[index];
  };

  // Return the type of an object
  function toType(v) {
    return ({}).toString.call(v).match(/\s([a-zA-Z]+)/)[1].toLowerCase();
  };

  // Getter/setter functions
  slider.min = function(_) {
    if (!arguments.length) return min;
    min = _;
    return slider;
  };

  slider.max = function(_) {
    if (!arguments.length) return max;
    max = _;
    return slider;
  };

  slider.step = function(_) {
    if (!arguments.length) return step;
    step = _;
    return slider;
  };

  slider.animate = function(_) {
    if (!arguments.length) return animate;
    animate = _;
    return slider;
  };

  slider.orientation = function(_) {
    if (!arguments.length) return orientation;
    orientation = _;
    return slider;
  };

  slider.axis = function(_) {
    if (!arguments.length) return axis;
    axis = _;
    return slider;
  };

  slider.margin = function(_) {
    if (!arguments.length) return margin;
    margin = _;
    return slider;
  };

  slider.value = function(_) {
    if (!arguments.length) return value;
    if (value) {
      moveHandle(stepValue(_));
    };
    value = _;
    return slider;
  };

  slider.snap = function(_) {
    if (!arguments.length) return snap;
    snap = _;
    return slider;
  };

  slider.scale = function(_) {
    if (!arguments.length) return scale;
    scale = _;
    return slider;
  };

  d3.rebind(slider, dispatch, "on");

  return slider;

}
}));