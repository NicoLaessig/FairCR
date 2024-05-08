import {Component, OnInit} from '@angular/core';
import {Chart, registerables} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import {HttpService} from '../../http.service'
import {SharedDataService} from "../../shared-data.service";
import _default from "chart.js/dist/plugins/plugin.tooltip";
import type = _default.defaults.animations.numbers.type;

// register all elements
Chart.register(...registerables);
Chart.register(zoomPlugin);

@Component({
  selector: 'app-dataset-chart',
  templateUrl: './dataset-chart.component.html',
  styleUrls: ['./dataset-chart.component.scss']
})
/**
 * This class represent the Scatter plot that visualizes the (dimension Reduced) Dataset
 */
export class DatasetChartComponent implements OnInit {
  chart: Chart | null = null;

  hasData: boolean = false;

  attributes: string[] = ["tsne"]

  attributeXAxis: string = "tsne"
  attributeYAxis: string = "tsne"

  constructor(private sharedDataService: SharedDataService, private httpService: HttpService) {

  }


  ngOnInit() {
    // subscribe to the data sharing service, so you know when the data changes
    this.sharedDataService.sharedData.subscribe(data => {
      // set the list of attributes
      this.attributes = [];
      this.attributes.push("tsne");
      this.attributes.push(...data["attributes"]);
      // When the data changes, call setDataPoints methode
      this.setDataPoints(data);
    });

    // configurate the chart:
    const config: any = {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: "label0",
            data: []
          },
          {
            label: "label1",
            data: []
          }
        ]
      },
      options: {

        /**
         * This Method defines what will happen when you click on a point in the chart.
         * It takes the point, sends it to the backend and receives the corresponding (multi-dimensional) datapoint.
         * Then it shares this new data with the table.
         * @param event
         */
        onClick: (event: Event) => {
          if (this.chart) {
            let point = this.chart.getElementsAtEventForMode(event, 'nearest', {intersect: false}, true);
            let dataset = point[0].datasetIndex;
            let index = point[0].index;
            let coordinates: any = this.chart.data.datasets[dataset].data[index];
            this.httpService.sendDatasetTSNEPoint(dataset, index, coordinates).subscribe(
              response => {
                // set the data in the Data Share Service
                let attributes: string[] = [];
                let values = [];
                const entries = Object.entries(response);
                for (const [key, value] of entries) {
                  attributes.push(key);
                  values.push(value);
                }
                this.sharedDataService.updateDataPoint([attributes, values]);


              }
            )
          }
        },
        // Interaction mode determines which point is shown in the toolbar when hovering over a point.
        // Nearest = the Nearest Point to the cursor
        interaction: {
          mode: 'nearest'
        },
        scales: {
          x: {
            type: 'linear',
            position: 'bottom'
          },
          y: {
            type: 'linear',
            position: 'left'
          }
        },
        plugins: {
          zoom: {
            zoom: {
              wheel: {
                enabled: true
              },
              pinch: {
                enabled: true
              },
              mode: 'xy',
            }
          },
          zoomActions: {
            enabled: true,
            actions: [
              {
                name: 'Reset Zoom',
                handler(chart: Chart) {
                  chart.resetZoom();
                },
              },
            ],
          }
        }

      }
    };

    // Create the chart
    this.chart = new Chart('scatter-chart', config);
  }

  setDataPoints(data: any) {
    console.log(this.chart)
    if (this.chart) {
      this.hasData = true;
      this.chart.data.datasets[0].data = data["label0"];
      this.chart.data.datasets[1].data = data["label1"];
      this.chart.update();
      // show the container , chart
      const container = document.getElementById('chartContainer');
      if (container) {
        container.style.visibility = 'visible';
      }
    }
  }

  resetZoom() {
    if (this.chart) {
      this.chart.resetZoom();
    }
    console.log(this.attributes)
  }

  setAxes() {
    //special case when one attribute is tsne and the other not -> TODO Raise Toast
    if (this.attributeXAxis == "tsne" && this.attributeYAxis != "tsne" || this.attributeXAxis != "tsne" && this.attributeYAxis == "tsne") {
      console.log("not possible")
    } else {
      this.httpService.sendDatasetAttributes(this.attributeXAxis, this.attributeYAxis).subscribe(
        response => {
          this.setDataPoints(response);
        }
      )
    }
  }
}

