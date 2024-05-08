import {Component} from '@angular/core';
import { HttpService } from '../../http.service'
import {SharedDataService} from "../../shared-data.service";

@Component({
  selector: 'app-input-field',
  templateUrl: './input-field.component.html',
  styleUrls: ['./input-field.component.scss']
})
export class InputFieldComponent {
  inputRows = [{
      dataset: 'communities',
      // sensAttrs: '',
      // favored: '',
      label: 'crime',
      // metric: '',
      // training: '',
      // proxy: '',
      // allowed: '',
      // ccr: '',
      // ca: '',
      randomstate: '-1'
    }];

  constructor(private httpService: HttpService, private sharedDataService: SharedDataService) { }

  /**
   * This functions adds a new input row (run)
   */
  addRow() {
    this.inputRows.push({
      dataset: '',
      // sensAttrs: '',
      // favored: '',
      label: '',
      // metric: '',
      // training: '',
      // proxy: '',
      // allowed: '',
      // ccr: '',
      // ca: '',
      randomstate: ''
    });
  }

  /**
   * This function delete a input row (run)
   */
  deleteRow() {
    this.inputRows.pop();
  }


  /**
   * This function is called when pressing the submit Button.
   * It takes the data from the different runs and sends it to the backend and waits for the response.
   *
   * !!! this can take very long !!!
   */
  submitData() {
    // Daten aus dem Eingabeformular abrufen
    let data = this.inputRows;
    this.httpService.sendDatasetInput(data).subscribe(
      (response: any) => {
        if (response && response.label0 && response.label1 && response.attributes) {
          // Die Daten, die mit dem Graphen-Component geteilt werden sollen, aktualisieren
          let attributes: string[] = response.attributes;
          let label0Data: number[] = response.label0;
          let label1Data: number[] = response.label1;
          this.sharedDataService.updateData({ "attributes" : attributes, "label0": label0Data, "label1": label1Data });
        } else {
          console.error("Ungültige Serverantwort: Erwartete Eigenschaften fehlen.");
        }
      },
      error => {
        console.error("Fehler beim Senden der Daten an den Server:", error);

      }
    );
  }
}
