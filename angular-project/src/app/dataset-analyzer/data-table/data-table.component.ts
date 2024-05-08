import {Component} from '@angular/core';
import {SharedDataService} from "../../shared-data.service";
import {HttpService} from "../../http.service";


@Component({
  selector: 'app-data-table',
  templateUrl: './data-table.component.html',
  styleUrls: ['./data-table.component.scss']
})
export class DataTableComponent {

  // These flags represent if the user want to select a (new) datapoint
  button1Active: boolean = false;
  button2Active: boolean = false;

  // These are the attributes + values for the selcted data point
  attributes: string[] = [];
  valuesPoint1: any[] = [];
  valuesPoint2: any[] = [];

  // Value only visible if both data points selcted:
  similarities: any = {euclidean: 0}

  constructor(private sharedDataService: SharedDataService, private httpService: HttpService) {
  }

  ngOnInit() {
    // subscribe to the data Sharing Service, so you know when someone selected a new point.
    // When someone selects a new point show the new data in the table
    this.sharedDataService.sharedDataPoint$.subscribe(newAttributes => {
      this.attributes = newAttributes[0];

      // depending on which datapoint we want to change
      if (this.button2Active) {
        this.valuesPoint2 = newAttributes[1];
      } else{
        this.valuesPoint1 = newAttributes[1]
      }
      // if the table got reset, then clear the table content
      // This special case occurs when changing the dataset you want to display
      if (this.attributes.length == 0){
        this.valuesPoint1 = [];
        this.valuesPoint2 = [];
      }

      // if both datapoints are selected calculate the similiarity
      if(this.valuesPoint1.length>0 && this.valuesPoint2.length>0){
        this.getSimiliarty();
      }

    });
  }

  activateButton(buttonNumber: number) {
    if (buttonNumber === 1) {
      this.button1Active = true;
      this.button2Active = false;
    } else {
      this.button1Active = false;
      this.button2Active = true;
    }
  }

  /**
   * This method is used to send the 2 datapoint to the backend and receive the
   * similiarty
   */
  getSimiliarty() {
    this.httpService.sendDatasetPoints(this.valuesPoint1, this.valuesPoint2).subscribe(
      response => {
        this.similarities = response;
      }
    )
  }

}
