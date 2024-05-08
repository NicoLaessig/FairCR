import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MetricBoxplotAccordionItemComponent } from './metric-boxplot-accordion-item.component';

describe('MetricBoxplotAccordionItemComponent', () => {
  let component: MetricBoxplotAccordionItemComponent;
  let fixture: ComponentFixture<MetricBoxplotAccordionItemComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [MetricBoxplotAccordionItemComponent]
    });
    fixture = TestBed.createComponent(MetricBoxplotAccordionItemComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
