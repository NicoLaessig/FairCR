import { Injectable } from '@angular/core';
import {BehaviorSubject} from "rxjs";

@Injectable({
  providedIn: 'root'
})
/**
 * This class is used to communicate between none related componentes.
 *
 * Example Use Case:
 * Component A wants to send data to Component B then he has to do the following steps:
 * 1. change the corresponding data in the sharedDataService
 * 2. Component B has to subscribe to the sharedDataService, so he knows when the data changes
 * 3. Component B realises the changes and pulls out the new data.
 */
export class SharedDataService {
  // Shared Data between the  Dataset analyzer Input Component and the  Dataset analyzer Table Component
  private data = new BehaviorSubject<{"attributes": string[], "label0":number[], "label1":number[]}>({"attributes": [], "label0" : [], "label1" : []})
  sharedData = this.data.asObservable();


  // Shared Data between the Dataset analyzer Graph Component and the  Dataset analyzer  Table Component
  private extendedDataPoint = new BehaviorSubject<[string[],any[]]>([[],[]])
  sharedDataPoint$ = this.extendedDataPoint.asObservable();

    /**
   * This Method updates the dataset shared between the input and the chart.
   * @param data the datapoints for eacth label
   */
  updateData(data:{"attributes":string[], "label0":number[], "label1":number[]}){
    this.data.next(data)
  }



  /**
   * This Method updates the data shared between the chart and the table.
   * @param data the attribute values pairs of the selected point.
   */
  updateDataPoint(data:[string[],any[]]){
    this.extendedDataPoint.next(data);
  }


 // ---------------------------------- Run Analyzer -------------------------------

 // Shared Data between the Run analyzer Input Component and the Run anylzer Chart Component
 private metricData = new BehaviorSubject<any>({});
  sharedMetricData$ = this.metricData.asObservable();

  updateMetricData(data: any){
    this.metricData.next(data);
  }


}
