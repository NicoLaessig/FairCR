import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MetricBarchartAccordionItemComponent } from './metric-barchart-accordion-item.component';

describe('MetricBarchartAccordionItemComponent', () => {
  let component: MetricBarchartAccordionItemComponent;
  let fixture: ComponentFixture<MetricBarchartAccordionItemComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [MetricBarchartAccordionItemComponent]
    });
    fixture = TestBed.createComponent(MetricBarchartAccordionItemComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
