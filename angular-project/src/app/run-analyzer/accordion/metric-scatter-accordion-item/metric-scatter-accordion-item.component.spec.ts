import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MetricScatterAccordionItemComponent } from './metric-scatter-accordion-item.component';

describe('MetricScatterAccordionItemComponent', () => {
  let component: MetricScatterAccordionItemComponent;
  let fixture: ComponentFixture<MetricScatterAccordionItemComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [MetricScatterAccordionItemComponent]
    });
    fixture = TestBed.createComponent(MetricScatterAccordionItemComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
